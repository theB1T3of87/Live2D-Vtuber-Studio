package com.foxgirlsupremacy.companion

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import com.example.mediapipefacelandmark.Camera2Manager
import com.example.mediapipefacelandmark.MediaPipeFaceDetector
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * VirtualFaceCapture-MNN camera pipeline adapter.
 * Camera2 supplies rotated RGB frames; MNN performs face detection, 478-point
 * landmark inference and ARKit-52 blendshape prediction.
 */
class MnnFaceTrackingController(
    context: Context,
    private val onTracking: (FaceTrackingData?) -> Unit,
    private val onStatus: (String) -> Unit = {},
) {
    private val appContext = context.applicationContext
    private val worker = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())
    private val processing = AtomicBoolean(false)
    @Volatile private var stopped = false
    private var camera: Camera2Manager? = null
    private var detector: MediaPipeFaceDetector? = null
    private var neutralPose: FloatArray? = null
    private var lastFaceAt = 0L

    private fun status(message: String) {
        if (!stopped) main.post { if (!stopped) onStatus(message) }
    }

    fun start() {
        status("Inicializando VirtualFaceCapture-MNN…")
        worker.execute {
            try {
                val faceDetector = MediaPipeFaceDetector(appContext)
                if (!faceDetector.initialize(numThreads = 2, enableBlendshape = true)) {
                    throw IllegalStateException("MNN não conseguiu carregar os modelos faciais")
                }
                if (stopped) {
                    faceDetector.release()
                    return@execute
                }
                detector = faceDetector
                val cameraManager = Camera2Manager(appContext)
                camera = cameraManager
                cameraManager.setFrameCallback { frame, width, height ->
                    if (!stopped && processing.compareAndSet(false, true)) {
                        worker.execute {
                            try {
                                processFrame(faceDetector, frame.duplicate(), width, height)
                            } catch (error: Exception) {
                                Log.e(TAG, "Falha ao processar quadro MNN", error)
                                publish(null)
                            } finally {
                                processing.set(false)
                            }
                        }
                    }
                }
                status("MNN pronto; abrindo câmera frontal…")
                cameraManager.startCamera()
            } catch (error: Exception) {
                Log.e(TAG, "Falha ao iniciar VirtualFaceCapture-MNN", error)
                status("Falha no rastreamento MNN: ${error.message ?: error.javaClass.simpleName}")
                publish(null)
            }
        }
    }

    private fun processFrame(
        faceDetector: MediaPipeFaceDetector,
        frame: ByteBuffer,
        width: Int,
        height: Int,
    ) {
        if (stopped) return
        val landmarks = faceDetector.detectLandmarks(frame, width, height)
        if (landmarks == null || landmarks.size < 468) {
            if (System.currentTimeMillis() - lastFaceAt > LOST_FACE_HOLD_MS) publish(null)
            status("Câmera ativa; procurando rosto…")
            return
        }
        val shapes = faceDetector.predictBlendshapes(landmarks, width, height, true)
        val pose = faceDetector.estimateHeadPose(landmarks, width, height)
        if (shapes == null || pose == null) {
            status("Rosto detectado; aguardando parâmetros faciais…")
            return
        }

        val reference = neutralPose ?: floatArrayOf(pose.first, pose.second, pose.third).also { neutralPose = it }
        val angleX = (pose.second - reference[1]).coerceIn(-30f, 30f)
        val angleY = (reference[0] - pose.first).coerceIn(-30f, 30f)
        val angleZ = (pose.third - reference[2]).coerceIn(-30f, 30f)
        val values = shapes.values
        val smile = ((values.getOrElse(44) { 0f } + values.getOrElse(45) { 0f }) * 0.5f * 2f - 1f)
            .coerceIn(-1f, 1f)
        val data = FaceTrackingData(
            angleX = angleX,
            angleY = angleY,
            angleZ = angleZ,
            eyeLOpen = (1f - shapes.eyeBlinkLeft).coerceIn(0f, 1f),
            eyeROpen = (1f - shapes.eyeBlinkRight).coerceIn(0f, 1f),
            mouthForm = smile,
            mouthOpen = shapes.jawOpen.coerceIn(0f, 1f),
            gazeX = 0f,
            gazeY = 0f,
        )
        lastFaceAt = System.currentTimeMillis()
        status("VirtualFaceCapture-MNN ativo; rosto rastreado")
        publish(data)
    }

    private fun publish(data: FaceTrackingData?) {
        if (!stopped) main.post { if (!stopped) onTracking(data) }
    }

    fun stop() {
        if (stopped) return
        stopped = true
        camera?.stopCamera()
        camera = null
        worker.execute {
            detector?.release()
            detector = null
        }
        worker.shutdown()
        main.post { onTracking(null) }
    }

    companion object {
        private const val TAG = "MnnFaceTracking"
        private const val LOST_FACE_HOLD_MS = 650L
    }
}
