package com.foxgirlsupremacy.companion

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import com.example.mediapipefacelandmark.Camera2Manager
import com.example.mediapipefacelandmark.MediaPipeFaceDetector
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs

/**
 * End-to-end camera -> MNN face mesh -> ARKit blendshapes -> Cubism tracking pipeline.
 *
 * Camera2Manager reuses its RGB output buffer. Frames are copied immediately in the
 * camera callback before being queued, so inference never reads memory that the
 * camera thread may already have overwritten.
 */
class MnnFaceTrackingController(
    context: Context,
    private val onTracking: (FaceTrackingData?) -> Unit,
    private val onStatus: (String) -> Unit = {},
) {
    private val appContext = context.applicationContext
    private val worker = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "Live2D-Face-Inference")
    }
    private val main = Handler(Looper.getMainLooper())
    private val processing = AtomicBoolean(false)

    @Volatile private var stopped = false
    @Volatile private var camera: Camera2Manager? = null
    @Volatile private var detector: MediaPipeFaceDetector? = null
    private var neutralSamples = ArrayList<FloatArray>(CALIBRATION_FRAMES)
    private var neutralPose: FloatArray? = null
    private var filtered: FaceTrackingData? = null
    private var lastFaceAtMs = 0L
    private var lastStatus = ""

    private fun status(message: String) {
        if (stopped || message == lastStatus) return
        lastStatus = message
        main.post { if (!stopped) onStatus(message) }
    }

    fun start() {
        if (stopped) return
        status("Preparando câmera e modelos faciais…")
        worker.execute {
            try {
                val faceDetector = MediaPipeFaceDetector(appContext)
                if (!faceDetector.initialize(numThreads = 2, enableBlendshape = true)) {
                    throw IllegalStateException("Não foi possível carregar os modelos MNN")
                }
                if (stopped) {
                    faceDetector.release()
                    return@execute
                }
                detector = faceDetector

                val cameraManager = Camera2Manager(appContext)
                camera = cameraManager
                cameraManager.setFrameCallback { reusableFrame, width, height ->
                    if (stopped || !processing.compareAndSet(false, true)) return@setFrameCallback

                    // The camera manager reuses this buffer for subsequent frames.
                    // Own the bytes before returning from its callback.
                    val ownedFrame = try {
                        val source = reusableFrame.duplicate()
                        val copy = ByteArray(source.remaining())
                        source.get(copy)
                        ByteBuffer.wrap(copy)
                    } catch (error: Exception) {
                        processing.set(false)
                        Log.e(TAG, "Não foi possível copiar o frame da câmera", error)
                        return@setFrameCallback
                    }

                    try {
                        worker.execute {
                            try {
                                if (!stopped) processFrame(faceDetector, ownedFrame, width, height)
                            } catch (error: Exception) {
                                Log.e(TAG, "Erro durante inferência facial", error)
                                status("Erro ao processar rosto: ${error.javaClass.simpleName}")
                            } finally {
                                processing.set(false)
                            }
                        }
                    } catch (error: Exception) {
                        processing.set(false)
                        if (!stopped) Log.e(TAG, "Fila de inferência indisponível", error)
                    }
                }

                status("Modelos carregados; iniciando câmera frontal…")
                cameraManager.startCamera()
            } catch (error: Exception) {
                Log.e(TAG, "Falha ao iniciar rastreamento facial", error)
                status("Falha no rastreamento: ${error.message ?: error.javaClass.simpleName}")
                publish(null)
                runCatching { detector?.release() }
                detector = null
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
        if (landmarks == null || landmarks.size < MIN_LANDMARKS) {
            if (lastFaceAtMs == 0L || SystemClock.elapsedRealtime() - lastFaceAtMs > LOST_FACE_HOLD_MS) {
                publish(null)
            }
            status("Câmera ativa; procurando rosto…")
            return
        }

        val shapes = faceDetector.predictBlendshapes(landmarks, width, height, true)
        val pose = faceDetector.estimateHeadPose(landmarks, width, height)
        if (shapes == null || pose == null || shapes.values.size < MediaPipeFaceDetector.BS_COUNT) {
            status("Rosto detectado; aguardando dados de expressão…")
            return
        }

        val currentPose = floatArrayOf(pose.first, pose.second, pose.third)
        val neutral = neutralPose
        if (neutral == null) {
            neutralSamples.add(currentPose)
            status("Rosto detectado; calibrando posição neutra…")
            if (neutralSamples.size >= CALIBRATION_FRAMES) {
                neutralPose = FloatArray(3) { axis ->
                    neutralSamples.sumOf { it[axis].toDouble() }.toFloat() / neutralSamples.size
                }
                neutralSamples.clear()
                status("Calibração concluída; rastreamento ativo")
            }
            return
        }

        val values = shapes.values
        fun blend(index: Int) = values.getOrElse(index) { 0f }.coerceIn(0f, 1f)

        // MediaPipe returns pitch, yaw, roll. Convert to Cubism's X/Y/Z axes.
        val angleX = (pose.second - neutral[1]).times(HEAD_YAW_GAIN).coerceIn(-30f, 30f)
        val angleY = (neutral[0] - pose.first).times(HEAD_PITCH_GAIN).coerceIn(-25f, 25f)
        val angleZ = (pose.third - neutral[2]).times(HEAD_ROLL_GAIN).coerceIn(-30f, 30f)

        // ARKit 52 ordering, using the named indices declared by MediaPipeFaceDetector.
        val gazeTowardSubjectRight = (blend(MediaPipeFaceDetector.BS_EYE_LOOK_IN_LEFT) +
            blend(MediaPipeFaceDetector.BS_EYE_LOOK_OUT_RIGHT)) * 0.5f
        val gazeTowardSubjectLeft = (blend(MediaPipeFaceDetector.BS_EYE_LOOK_OUT_LEFT) +
            blend(MediaPipeFaceDetector.BS_EYE_LOOK_IN_RIGHT)) * 0.5f
        val gazeX = ((gazeTowardSubjectLeft - gazeTowardSubjectRight) * GAZE_GAIN).coerceIn(-1f, 1f)
        val gazeY = ((blend(MediaPipeFaceDetector.BS_EYE_LOOK_UP_LEFT) +
            blend(MediaPipeFaceDetector.BS_EYE_LOOK_UP_RIGHT)) -
            (blend(MediaPipeFaceDetector.BS_EYE_LOOK_DOWN_LEFT) +
            blend(MediaPipeFaceDetector.BS_EYE_LOOK_DOWN_RIGHT))) * GAZE_VERTICAL_GAIN

        val smile = (blend(MediaPipeFaceDetector.BS_MOUTH_SMILE_LEFT) +
            blend(MediaPipeFaceDetector.BS_MOUTH_SMILE_RIGHT)) * 0.5f
        val frown = (blend(MediaPipeFaceDetector.BS_MOUTH_FROWN_LEFT) +
            blend(MediaPipeFaceDetector.BS_MOUTH_FROWN_RIGHT)) * 0.5f

        val target = FaceTrackingData(
            angleX = angleX,
            angleY = angleY,
            angleZ = angleZ,
            eyeLOpen = (1f - shapes.eyeBlinkLeft).coerceIn(0f, 1f),
            eyeROpen = (1f - shapes.eyeBlinkRight).coerceIn(0f, 1f),
            mouthForm = ((smile - frown) * MOUTH_FORM_GAIN).coerceIn(-1f, 1f),
            mouthOpen = ((shapes.jawOpen - JAW_OPEN_DEAD_ZONE) / JAW_OPEN_RESPONSE).coerceIn(0f, 1f),
            gazeX = gazeX,
            gazeY = gazeY.coerceIn(-1f, 1f),
        )

        lastFaceAtMs = SystemClock.elapsedRealtime()
        val stable = smooth(filtered, target)
        filtered = stable
        status("Rosto detectado; rastreamento ativo")
        publish(stable)
        maybeLog(stable)
    }

    private fun smooth(previous: FaceTrackingData?, target: FaceTrackingData): FaceTrackingData {
        if (previous == null) return target
        fun lerp(from: Float, to: Float, amount: Float) = from + (to - from) * amount
        fun eye(from: Float, to: Float): Float {
            val targetValue = if (to > 0.96f) 1f else to
            return lerp(from, targetValue, if (targetValue < from) 0.78f else 0.42f)
        }
        return FaceTrackingData(
            angleX = lerp(previous.angleX, target.angleX, HEAD_SMOOTHING),
            angleY = lerp(previous.angleY, target.angleY, HEAD_SMOOTHING),
            angleZ = lerp(previous.angleZ, target.angleZ, HEAD_SMOOTHING),
            eyeLOpen = eye(previous.eyeLOpen, target.eyeLOpen),
            eyeROpen = eye(previous.eyeROpen, target.eyeROpen),
            mouthForm = lerp(previous.mouthForm, target.mouthForm, EXPRESSION_SMOOTHING),
            mouthOpen = lerp(previous.mouthOpen, target.mouthOpen, MOUTH_SMOOTHING),
            gazeX = lerp(previous.gazeX, target.gazeX, GAZE_SMOOTHING),
            gazeY = lerp(previous.gazeY, target.gazeY, GAZE_SMOOTHING),
        )
    }

    private fun maybeLog(data: FaceTrackingData) {
        val now = SystemClock.elapsedRealtime()
        if (now - lastDiagnosticAtMs < DIAGNOSTIC_INTERVAL_MS) return
        lastDiagnosticAtMs = now
        Log.i(TAG, "Cubism input angle=(%.1f,%.1f,%.1f) eyes=(%.2f,%.2f) mouth=(%.2f,%.2f) gaze=(%.2f,%.2f)".format(
            data.angleX, data.angleY, data.angleZ, data.eyeLOpen, data.eyeROpen,
            data.mouthForm, data.mouthOpen, data.gazeX, data.gazeY,
        ))
    }

    private var lastDiagnosticAtMs = 0L

    private fun publish(data: FaceTrackingData?) {
        if (!stopped) main.post { if (!stopped) onTracking(data) }
    }

    fun stop() {
        if (stopped) return
        stopped = true
        runCatching { camera?.stopCamera() }
        camera = null
        // Release MNN on its inference thread after any in-flight inference has finished.
        runCatching {
            worker.execute {
                runCatching { detector?.release() }
                detector = null
            }
        }
        worker.shutdown()
        neutralSamples.clear()
        neutralPose = null
        filtered = null
        main.post { onTracking(null) }
    }

    companion object {
        private const val TAG = "Live2DFacePipeline"
        private const val MIN_LANDMARKS = 468
        private const val CALIBRATION_FRAMES = 24
        private const val LOST_FACE_HOLD_MS = 650L
        private const val DIAGNOSTIC_INTERVAL_MS = 2_000L
        private const val HEAD_YAW_GAIN = 1.15f
        private const val HEAD_PITCH_GAIN = 1.25f
        private const val HEAD_ROLL_GAIN = 1.0f
        private const val GAZE_GAIN = 1.8f
        private const val GAZE_VERTICAL_GAIN = 0.9f
        private const val MOUTH_FORM_GAIN = 1.6f
        private const val JAW_OPEN_DEAD_ZONE = 0.08f
        private const val JAW_OPEN_RESPONSE = 0.82f
        private const val HEAD_SMOOTHING = 0.38f
        private const val EXPRESSION_SMOOTHING = 0.42f
        private const val MOUTH_SMOOTHING = 0.62f
        private const val GAZE_SMOOTHING = 0.5f
    }
}
