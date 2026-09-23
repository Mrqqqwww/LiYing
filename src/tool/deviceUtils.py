import onnxruntime as ort

def is_cuda_really_available():
    try:
        available_providers = ort.get_available_providers()
        return "CUDAExecutionProvider" in available_providers
    except:
        return False

ONNX_DEVICE = "GPU" if is_cuda_really_available() else "CPU"
ONNX_PROVIDER = "CUDAExecutionProvider" if ONNX_DEVICE == "GPU" else "CPUExecutionProvider"

def get_onnx_session(model_path: str):
    """
    Creates an ONNX inference session, automatically selecting the GPU if available, otherwise defaulting to the CPU.
    
    Args:
        model_path: The file path to the ONNX model.
        
    Returns:
        ort.InferenceSession: The created ONNX inference session.
        
    Notes:
        - The GPU will be prioritized for inference if CUDA is supported and onnxruntime-gpu is installed.
        - Automatically falls back to the CPU if loading on the GPU fails.
    """
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ONNX_PROVIDER == "CUDAExecutionProvider"
        else ["CPUExecutionProvider"]
    )
    print(f"Using {ONNX_DEVICE} for inference with model: {model_path}")

    sess_opts = None
    import os
    if os.environ.get("LIYING_ORT_NOARENA"):
        sess_opts = ort.SessionOptions()
        sess_opts.enable_cpu_mem_arena = False
        sess_opts.enable_mem_pattern = False
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1

    try:
        if ONNX_DEVICE == "GPU":
            print(f"Attempting to load the model '{model_path}' using CUDA")
        if sess_opts is not None:
            return ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
        return ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        if ONNX_PROVIDER == "CUDAExecutionProvider":
            print(f"Failed to load model '{model_path}' on GPU: {e}")
            print(f"Falling back to CPU for inference with model: {model_path}")
            return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        raise RuntimeError(f"Failed to load the ONNX model: {e}")