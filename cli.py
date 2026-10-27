"""
Interactive Command-Line Interface (CLI) for Traffic Sign Model Evaluation.

Dynamically scans the models/ directory, presents an interactive menu,
and delegates evaluation to eval.py.
"""

import os
import argparse
import importlib
import inspect
import torch

from eval import Evaluator


def discover_available_models(models_dir='models'):
    """
    Scans the models/ directory dynamically for any subfolder containing a model.py file.
    
    Returns:
        dict: Mapping of model_name -> directory path
    """
    if not os.path.exists(models_dir):
        return {}

    discovered = {}
    for entry in sorted(os.listdir(models_dir)):
        entry_path = os.path.join(models_dir, entry)
        model_file = os.path.join(entry_path, 'model.py')
        if os.path.isdir(entry_path) and os.path.isfile(model_file) and not entry.startswith('.'):
            discovered[entry] = entry_path
    return discovered


def load_model_class(model_name, models_dir='models'):
    """
    Dynamically imports the model class from models/<model_name>/model.py.
    Finds the custom torch.nn.Module subclass defined inside the module.
    """
    module_path = f"models.{model_name}.model"
    module = importlib.import_module(module_path)

    # Find custom nn.Module subclasses defined in that specific module
    candidates = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (inspect.isclass(attr) 
                and issubclass(attr, torch.nn.Module) 
                and getattr(attr, '__module__', None) == module.__name__):
            if attr not in candidates:
                candidates.append(attr)

    if not candidates:
        raise ImportError(f"No torch.nn.Module subclass found in '{module_path}'.")

    # Match preference: class name matching directory name (case-insensitive)
    for cls in candidates:
        if cls.__name__.lower() == model_name.lower():
            return cls
        if f"{model_name.lower()}model" == cls.__name__.lower():
            return cls

    # Default to first custom nn.Module subclass defined in the module
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Traffic Sign Recognition Models (Clean + Robustness Benchmark)")
    parser.add_argument('--model', type=str, default=None,
                        help="Model name to evaluate (optional; if omitted, an interactive prompt is shown).")
    parser.add_argument('--dataset', type=str, default='gtsrb', choices=['gtsrb', 'btsd', 'ctsd'],
                        help="Dataset benchmark to evaluate: 'gtsrb' (default: 43 classes), 'btsd' (62 classes), or 'ctsd' (58 classes).")
    parser.add_argument('--data-dir', type=str, default='data',
                        help="Root directory for dataset files (default: 'data').")
    parser.add_argument('--target-size', type=int, default=49,
                        help="Square image resolution for preprocessing (default: 49).")
    parser.add_argument('--batch-size', type=int, default=64,
                        help="Batch size for training and inference (default: 64).")
    parser.add_argument('--workers', type=int, default=4,
                        help="Number of DataLoader worker processes for parallel noise generation (default: 4).")
    parser.add_argument('--epochs', type=int, default=5,
                        help="Training epochs if model weights do not already exist (default: 5).")
    parser.add_argument('--device', type=str, default=None,
                        help="Computation device ('cuda' or 'cpu'). Auto-detected if omitted.")
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Custom output directory. Defaults to 'output/<model_name>'.")
    parser.add_argument('--weights', type=str, default=None,
                        help="Explicit checkpoint file path to load.")
    args = parser.parse_args()

    # Discover models dynamically from the directory structure
    available_models = discover_available_models('models')
    if not available_models:
        print("[Error] No models found in 'models/' directory. Expected layout: models/<model_name>/model.py")
        exit(1)

    # Resolve model selection: from --model flag or interactive prompt
    sorted_model_names = sorted(available_models.keys(), key=lambda x: (x != 'baseline', x))
    default_model = 'baseline' if 'baseline' in sorted_model_names else sorted_model_names[0]

    selected_name = None
    if args.model:
        m_lower = args.model.strip().lower()
        matched = [m for m in sorted_model_names if m.lower() == m_lower]
        if matched:
            selected_name = matched[0]
        else:
            print(f"[Warning] Model '{args.model}' not found in 'models/'. Discovered: {sorted_model_names}")
    
    if selected_name is None:
        print("\n" + "="*50)
        print(" Traffic Sign Recognition Model Evaluation")
        print("="*50)
        print(" Discovered models in 'models/' directory:")
        for idx, model_name in enumerate(sorted_model_names, 1):
            print(f"   [{idx}] {model_name}")
        print("="*50)

        try:
            user_choice = input(f"Enter model to evaluate (number or name) [default: {default_model}]: ").strip().lower()
        except EOFError:
            user_choice = default_model
            print(default_model)

        selected_name = default_model
        if user_choice:
            if user_choice.isdigit() and 1 <= int(user_choice) <= len(sorted_model_names):
                selected_name = sorted_model_names[int(user_choice) - 1]
            elif user_choice in sorted_model_names:
                selected_name = user_choice
            else:
                matched = [m for m in sorted_model_names if m.lower() == user_choice]
                if matched:
                    selected_name = matched[0]
                else:
                    print(f"[Warning] Unknown model '{user_choice}'. Falling back to default: '{default_model}'.")
                    selected_name = default_model

    # Dynamically load the selected class from its folder
    selected_model_cls = load_model_class(selected_name, 'models')
    print(f"\n--> Loaded class '{selected_model_cls.__name__}' dynamically from models/{selected_name}/model.py")
    print(f"--> Selected model: {selected_name.upper()}")
    print(f"--> Selected dataset: {args.dataset.upper()}\n")

    # Initialize evaluator (eval handles hardware detection & dataset loading)
    evaluator = Evaluator(
        data_dir=args.data_dir,
        target_size=(args.target_size, args.target_size),
        device=args.device,
        num_workers=args.workers
    )

    # Run evaluation
    evaluator.eval_model(
        model_cls_or_instance=selected_model_cls,
        model_name=selected_name,
        dataset=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.workers,
        weights_path=args.weights,
        load_if_exists=True
    )


if __name__ == "__main__":
    main()
