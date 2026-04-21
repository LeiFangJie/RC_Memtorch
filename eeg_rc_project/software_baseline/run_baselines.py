"""
软件基线模型的统一入口点。
支持运行单个模型或全部模型，支持单患者或全量数据模式。

用法:
    python run_baselines.py --model cnn1d --patient chb01
    python run_baselines.py --model all --patient chb01
    python run_baselines.py --model cnn1d --mode all
"""
import os
import sys
import argparse

# 添加父目录到路径以导入config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline_models import (
    Baseline_CNN1D, 
    Baseline_CNN_LSTM, 
    Baseline_Transformer, 
    Baseline_DeepCNN,
    RC_CNN,
    count_parameters
)
from baseline_trainer import train_baseline, save_results_to_csv
from band_decomposition import process_patient_bands
from config import ALL_PATIENTS, PROCESSED_DATA_DIR


# 模型名称到类和名称的映射
MODEL_REGISTRY = {
    "cnn1d": (Baseline_CNN1D, "Baseline_CNN1D"),
    "cnn_lstm": (Baseline_CNN_LSTM, "Baseline_CNN_LSTM"),
    "transformer": (Baseline_Transformer, "Baseline_Transformer"),
    "deepcnn": (Baseline_DeepCNN, "Baseline_DeepCNN"),
    "rc_cnn": (RC_CNN, "RC_CNN"),  # RC+CNN 架构（RC物理层 + 轻量级CNN分类器）
}


# 模型默认超参数配置 - 可在代码开头自定义每个模型的epochs等参数
MODEL_CONFIG = {
    "cnn1d": {
        "epochs": 20,
        "lr": 0.0005,
        "batch_size": 256,
        "weight_decay": 1e-4,
    },
    "cnn_lstm": {
        "epochs": 80,      # LSTM需要更多epochs收敛
        "lr": 0.0003,
        "batch_size": 128,
        "weight_decay": 1e-4,
    },
    "transformer": {
        "epochs": 80,      # Transformer需要更多epochs
        "lr": 0.0001,      # 更小的学习率稳定训练
        "batch_size": 64,  # 更大的模型用更小batch
        "weight_decay": 1e-4,
    },
    "deepcnn": {
        "epochs": 50,
        "lr": 0.0005,
        "batch_size": 256,
        "weight_decay": 1e-4,
    },
    "rc_cnn": {
        "epochs": 50,      # 与主流程一致
        "lr": 0.0005,      # 与主流程一致
        "batch_size": 256, # 与主流程一致
        "weight_decay": 1e-4,
    },
}


def check_bands_data(patient_id):
    """检查频带分解数据是否存在"""
    bands_path = os.path.join(PROCESSED_DATA_DIR, f"{patient_id}_bands.pt")
    return os.path.exists(bands_path)


def prepare_data(patient_ids):
    """
    确保所有患者的频带分解数据已准备好。
    如果数据不存在，自动运行频带分解。
    """
    for pid in patient_ids:
        if not check_bands_data(pid):
            print(f"[Prepare] Bands data missing for {pid}, generating...")
            process_patient_bands(pid)
        else:
            print(f"[Prepare] Bands data ready for {pid}")


def get_available_patients():
    """获取已有频带分解数据的患者列表"""
    available = []
    for f in os.listdir(PROCESSED_DATA_DIR):
        if f.endswith('_bands.pt'):
            pid = f.replace('_bands.pt', '')
            available.append(pid)
    return sorted(available)


def run_single_model(model_key, patient_id, epochs=None, lr=None, 
                     batch_size=None, weight_decay=None, device=None):
    """
    运行单个基线模型。
    
    参数:
        model_key: 模型标识 (cnn1d/cnn_lstm/transformer/deepcnn)
        patient_id: 患者ID
        epochs: 训练轮数（默认从MODEL_CONFIG读取）
        lr: 学习率（默认从MODEL_CONFIG读取）
        batch_size: 批次大小（默认从MODEL_CONFIG读取）
        weight_decay: 权重衰减（默认从MODEL_CONFIG读取）
        device: 计算设备
        
    返回:
        result: 训练结果字典
    """
    if model_key not in MODEL_REGISTRY:
        print(f"[Error] Unknown model: {model_key}")
        print(f"Available models: {list(MODEL_REGISTRY.keys())}")
        return None
    
    model_class, model_name = MODEL_REGISTRY[model_key]
    
    # 从MODEL_CONFIG获取默认配置，参数传入则覆盖
    config = MODEL_CONFIG.get(model_key, {})
    epochs = epochs if epochs is not None else config.get("epochs", 15)
    lr = lr if lr is not None else config.get("lr", 0.0005)
    batch_size = batch_size if batch_size is not None else config.get("batch_size", 256)
    weight_decay = weight_decay if weight_decay is not None else config.get("weight_decay", 1e-4)
    
    print(f"[Config] {model_name}: epochs={epochs}, lr={lr}, batch_size={batch_size}")
    
    # 检查数据是否存在
    if not check_bands_data(patient_id):
        print(f"[Error] Bands data not found for {patient_id}")
        print(f"Run: python band_decomposition.py --patient {patient_id}")
        return None
    
    # 训练模型
    result = train_baseline(
        model_class=model_class,
        model_name=model_name,
        patient_id=patient_id,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        device=device
    )
    
    return result


def run_all_models(patient_id, epochs=None, lr=None, batch_size=None, 
                    weight_decay=None, device=None):
    """
    运行所有基线模型。
    
    参数:
        patient_id: 患者ID
        epochs: 训练轮数
        device: 计算设备
        
    返回:
        results: 结果字典列表
    """
    results = []
    
    for model_key in MODEL_REGISTRY.keys():
        # 不传具体参数，让每个模型使用MODEL_CONFIG中的默认配置
        result = run_single_model(model_key, patient_id, None, None, None, None, device)
        if result:
            results.append(result)
        print("\n" + "="*60 + "\n")
    
    return results


def run_multi_patient(model_key, patient_ids, epochs=None, lr=None, 
                      batch_size=None, weight_decay=None, device=None):
    """
    在多个患者上运行单个模型。
    
    参数:
        model_key: 模型标识
        patient_ids: 患者ID列表
        epochs: 训练轮数（默认从MODEL_CONFIG读取）
        lr: 学习率（默认从MODEL_CONFIG读取）
        batch_size: 批次大小（默认从MODEL_CONFIG读取）
        weight_decay: 权重衰减（默认从MODEL_CONFIG读取）
        device: 计算设备
        
    返回:
        all_results: 所有患者的结果列表
    """
    all_results = []
    
    for patient_id in patient_ids:
        print(f"\n{'#'*60}")
        print(f"Processing patient: {patient_id}")
        print(f"{'#'*60}")
        
        result = run_single_model(model_key, patient_id, epochs, lr, batch_size, weight_decay, device)
        if result:
            all_results.append(result)
    
    return all_results


def print_summary(results):
    """打印结果汇总表"""
    if not results:
        print("[Summary] No results to display.")
        return
    
    print("\n" + "="*80)
    print("BASELINE MODELS COMPARISON SUMMARY")
    print("="*80)
    
    # 表头
    header = f"{'Model':<20} {'Params':>12} {'FLOPs':>12} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Time(s)':>10}"
    print(header)
    print("-"*80)
    
    # 数据行
    for r in results:
        flops_str = f"{r['flops']/1e6:.1f}M" if r.get('flops') else "N/A"
        row = (f"{r['model_name']:<20} "
               f"{r['params']:>12,} "
               f"{flops_str:>12} "
               f"{r['f1']:>8.4f} "
               f"{r['precision']:>10.4f} "
               f"{r['recall']:>8.4f} "
               f"{r['training_time']:>10.1f}")
        print(row)
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Run software baseline models for EEG seizure detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single model on single patient
  python run_baselines.py --model cnn1d --patient chb01
  
  # Run all models on single patient
  python run_baselines.py --model all --patient chb01
  
  # Run specific model on all available patients
  python run_baselines.py --model transformer --mode all
  
  # Run with custom epochs (override MODEL_CONFIG)
  python run_baselines.py --model cnn_lstm --patient chb01 --epochs 20
  
  # Run with custom hyperparameters
  python run_baselines.py --model transformer --patient chb01 --epochs 30 --lr 0.0001 --batch-size 32
  
  # Use model-specific defaults from MODEL_CONFIG
  python run_baselines.py --model all --patient chb01
        """
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=["cnn1d", "cnn_lstm", "transformer", "deepcnn", "rc_cnn", "all"],
        help="Model to run"
    )
    
    parser.add_argument(
        "--patient", 
        type=str, 
        default="chb01",
        help="Patient ID (default: chb01)"
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["single", "all"],
        default="single",
        help="Run mode: single patient or all available patients"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None,
        help="Number of training epochs (default: from MODEL_CONFIG)"
    )
    
    parser.add_argument(
        "--lr", 
        type=float, 
        default=None,
        help="Learning rate (default: from MODEL_CONFIG)"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=None,
        help="Batch size (default: from MODEL_CONFIG)"
    )
    
    parser.add_argument(
        "--weight-decay", 
        type=float, 
        default=None,
        help="Weight decay (default: from MODEL_CONFIG)"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not specified."
    )
    
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Prepare band decomposition data before training"
    )
    
    args = parser.parse_args()
    
    # 确定设备
    if args.device:
        device = args.device
    else:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[Config] Device: {device}")
    if args.epochs:
        print(f"[Config] Override epochs: {args.epochs}")
    if args.lr:
        print(f"[Config] Override lr: {args.lr}")
    if args.batch_size:
        print(f"[Config] Override batch_size: {args.batch_size}")
    if args.weight_decay:
        print(f"[Config] Override weight_decay: {args.weight_decay}")
    print(f"[Info] Using MODEL_CONFIG defaults if not overridden")
    
    # 准备数据
    if args.prepare_data:
        if args.mode == "all":
            prepare_data(ALL_PATIENTS)
        else:
            prepare_data([args.patient])
    
    # 确定患者列表
    if args.mode == "all":
        patient_ids = get_available_patients()
        if not patient_ids:
            print("[Error] No band decomposition data found.")
            print("Run with --prepare-data to generate data.")
            return
        print(f"[Config] Running on {len(patient_ids)} patients: {patient_ids}")
    else:
        patient_ids = [args.patient]
    
    # 运行模型
    all_results = []
    
    if args.model == "all":
        # 在所有患者上运行所有模型（每个模型使用各自的MODEL_CONFIG配置）
        for patient_id in patient_ids:
            results = run_all_models(patient_id, args.epochs, args.lr, args.batch_size, args.weight_decay, device)
            all_results.extend(results)
    else:
        # 在指定患者上运行单个模型
        for patient_id in patient_ids:
            result = run_single_model(args.model, patient_id, args.epochs, args.lr, args.batch_size, args.weight_decay, device)
            if result:
                all_results.append(result)
    
    # 打印汇总
    print_summary(all_results)
    
    # 保存结果
    if all_results:
        save_results_to_csv(all_results)
        print("\n[Done] All results saved to software_baseline/results_comparison.csv")


if __name__ == "__main__":
    main()
