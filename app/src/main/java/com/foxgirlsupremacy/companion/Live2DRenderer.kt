package com.foxgirlsupremacy.companion

import android.content.Context
import android.graphics.SurfaceTexture
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.util.Log
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.locks.LockSupport

private const val FRAME_INTERVAL_NS = 16_666_667L
private const val FPS_LOG_INTERVAL_NS = 5_000_000_000L
private const val RENDER_TAG = "Live2DRenderer"
private const val MOTION_PRIORITY = 3

/** TextureView/EGL host for the Live2D Native SDK renderer. */
class Live2DRenderer(
    private val appContext: Context,
    private val modelDir: String,
    private val modelJson: String,
    private val onReady: () -> Unit = {},
) : CharacterRenderer {
    private var view: NativeTextureView? = null

    override fun createView(context: Context): View {
        NativeCubismBridge.configure(appContext, modelDir, modelJson)
        return NativeTextureView(context, onReady).also { view = it }
    }

    override fun setState(state: CharacterState) {
        if (state == CharacterState.PET || state == CharacterState.EAT || state == CharacterState.TALK) {
            view?.postNative { NativeCubismBridge.nativePlayMotion("TapBody", 0) }
        }
    }

    override fun playMotion(group: String, index: Int) {
        view?.postNative { NativeCubismBridge.nativePlayMotion(group, index) }
    }

    override fun playExpression(name: String) {
        view?.postNative { NativeCubismBridge.nativeSetExpression(name) }
    }

    fun setFaceTrackingData(data: FaceTrackingData?) {
        if (data == null) {
            NativeCubismBridge.nativeSetFaceTrackingData(false, 0f, 0f, 0f, 1f, 1f, 0f, 0f, 0f, 0f)
        } else {
            NativeCubismBridge.nativeSetFaceTrackingData(
                true, data.angleX, data.angleY, data.angleZ,
                data.eyeLOpen, data.eyeROpen, data.mouthForm, data.mouthOpen, data.gazeX, data.gazeY,
            )
        }
    }

    override fun setScale(scale: Float) = Unit
    override fun setAnimationsEnabled(enabled: Boolean) = Unit
    override fun setFacing(facing: Int) = Unit
    override fun update(dtMillis: Long) = Unit

    override fun dispose() {
        view?.stop()
        view = null
    }

    private class NativeTextureView(context: Context, private val onReady: () -> Unit) : TextureView(context), TextureView.SurfaceTextureListener {
        private var renderThread: NativeRenderThread? = null

        init {
            isOpaque = false
            surfaceTextureListener = this
        }

        fun postNative(action: () -> Unit) { renderThread?.post(action) }
        fun stop() { renderThread?.shutdown(); renderThread = null }

        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
            renderThread = NativeRenderThread(surface, width, height, onReady).also { it.start() }
        }
        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {
            renderThread?.resize(width, height)
        }
        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
            stop()
            return true
        }
        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) = Unit
    }

    private class NativeRenderThread(
        private val texture: SurfaceTexture,
        width: Int,
        height: Int,
        private val onReady: () -> Unit,
    ) : Thread("Fox-Cubism-GL") {
        @Volatile private var running = true
        @Volatile private var targetWidth = width
        @Volatile private var targetHeight = height
        @Volatile private var resizePending = true
        private val commands = ConcurrentLinkedQueue<() -> Unit>()
        private var display: EGLDisplay = EGL14.EGL_NO_DISPLAY
        private var context: EGLContext = EGL14.EGL_NO_CONTEXT
        private var eglSurface: EGLSurface = EGL14.EGL_NO_SURFACE
        private var androidSurface: Surface? = null

        fun post(action: () -> Unit) { commands.add(action) }
        fun resize(width: Int, height: Int) {
            targetWidth = width; targetHeight = height; resizePending = true
        }
        fun shutdown() {
            running = false
            interrupt()
            try { join(1500) } catch (_: InterruptedException) { Thread.currentThread().interrupt() }
        }

        override fun run() {
            if (!initEgl()) { releaseEgl(); return }
            try {
                NativeCubismBridge.nativeOnStart()
                NativeCubismBridge.nativeOnSurfaceCreated()
                NativeCubismBridge.nativeOnSurfaceChanged(targetWidth, targetHeight)
                onReady()
                // Pace against an absolute deadline. Sleeping 16 ms AFTER drawing
                // added the full render/swap cost to every frame (e.g. 12 ms render +
                // 16 ms sleep capped the renderer near 36 FPS). This loop only waits
                // for the remainder of the 60 Hz budget and never adds a fixed delay
                // on top of GPU work or eglSwapBuffers' own vsync wait.
                var nextFrameNs = System.nanoTime()
                var fpsWindowStartNs = nextFrameNs
                var framesInWindow = 0
                var accumulatedFrameNs = 0L
                while (running) {
                    val frameStartNs = System.nanoTime()
                    while (true) (commands.poll() ?: break).invoke()
                    if (resizePending) {
                        NativeCubismBridge.nativeOnSurfaceChanged(targetWidth, targetHeight)
                        resizePending = false
                    }
                    NativeCubismBridge.nativeOnDrawFrame()
                    EGL14.eglSwapBuffers(display, eglSurface)
                    val frameEndNs = System.nanoTime()

                    framesInWindow++
                    accumulatedFrameNs += frameEndNs - frameStartNs
                    if (frameEndNs - fpsWindowStartNs >= FPS_LOG_INTERVAL_NS) {
                        val elapsedSeconds = (frameEndNs - fpsWindowStartNs) / 1_000_000_000.0
                        val fps = framesInWindow / elapsedSeconds
                        val averageFrameMs = accumulatedFrameNs / framesInWindow / 1_000_000.0
                        Log.i(RENDER_TAG, "Cubism render: %.1f FPS, average draw+swap %.2f ms".format(fps, averageFrameMs))
                        fpsWindowStartNs = frameEndNs
                        framesInWindow = 0
                        accumulatedFrameNs = 0L
                    }

                    nextFrameNs += FRAME_INTERVAL_NS
                    val remainingNs = nextFrameNs - System.nanoTime()
                    if (remainingNs > 0L) {
                        LockSupport.parkNanos(remainingNs)
                    } else if (-remainingNs >= FRAME_INTERVAL_NS) {
                        // Do not accumulate timing debt after a slow frame.
                        nextFrameNs = System.nanoTime()
                    }
                    if (Thread.interrupted() && !running) break
                }
            } catch (_: Throwable) {
                // Keep renderer failures isolated from the foreground overlay service.
            } finally {
                runCatching { NativeCubismBridge.nativeOnStop() }
                runCatching { NativeCubismBridge.nativeOnDestroy() }
                releaseEgl()
            }
        }

        private fun initEgl(): Boolean {
            display = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
            if (display == EGL14.EGL_NO_DISPLAY) return false
            val versions = IntArray(2)
            if (!EGL14.eglInitialize(display, versions, 0, versions, 1)) return false
            val attributes = intArrayOf(
                EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
                EGL14.EGL_SURFACE_TYPE, EGL14.EGL_WINDOW_BIT,
                EGL14.EGL_RED_SIZE, 8, EGL14.EGL_GREEN_SIZE, 8,
                EGL14.EGL_BLUE_SIZE, 8, EGL14.EGL_ALPHA_SIZE, 8,
                EGL14.EGL_NONE,
            )
            val configs = arrayOfNulls<EGLConfig>(1)
            val count = IntArray(1)
            if (!EGL14.eglChooseConfig(display, attributes, 0, configs, 0, 1, count, 0) || count[0] == 0) return false
            val contextAttributes = intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE)
            context = EGL14.eglCreateContext(display, configs[0], EGL14.EGL_NO_CONTEXT, contextAttributes, 0)
            val surfaceAttributes = intArrayOf(EGL14.EGL_NONE)
            androidSurface = Surface(texture)
            eglSurface = EGL14.eglCreateWindowSurface(display, configs[0], androidSurface, surfaceAttributes, 0)
            if (context == EGL14.EGL_NO_CONTEXT || eglSurface == EGL14.EGL_NO_SURFACE) return false
            return EGL14.eglMakeCurrent(display, eglSurface, eglSurface, context)
        }

        private fun releaseEgl() {
            if (display != EGL14.EGL_NO_DISPLAY) {
                EGL14.eglMakeCurrent(display, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
                if (eglSurface != EGL14.EGL_NO_SURFACE) EGL14.eglDestroySurface(display, eglSurface)
                if (context != EGL14.EGL_NO_CONTEXT) EGL14.eglDestroyContext(display, context)
                EGL14.eglTerminate(display)
            }
            display = EGL14.EGL_NO_DISPLAY
            context = EGL14.EGL_NO_CONTEXT
            eglSurface = EGL14.EGL_NO_SURFACE
            androidSurface?.release()
            androidSurface = null
        }
    }
}
