"""
Interactive Command-Line Interface (CLI) for GTSRB Model Evaluation.

Dynamically scans the models/ directory, presents an interactive menu,
and orchestrates training and robustness benchmarking via eval.py.
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
    parser = argparse.ArgumentParser(description="Evaluate GTSRB Models (Clean + Robustness Benchmark)")
    parser.add_argument('--workers', type=int, default=None,
                        help="Number of DataLoader worker processes for parallel noise generation (default: auto).")
    parser.add_argument('--epochs', type=int, default=5,
                        help="Training epochs if model weights do not already exist (default: 5).")
    args = parser.parse_args()

    # Discover models dynamically from the directory structure
    available_models = discover_available_models('models')
    if not available_models:
        print("[Error] No models found in 'models/' directory. Expected layout: models/<model_name>/model.py")
        exit(1)

    print("\n" + "="*50)
    print(" GTSRB Model Evaluation Pipeline")
    print("="*50)
    print(" Discovered models in 'models/' directory:")
    
    # Sort with 'baseline' first if present, followed by others alphabetically
    sorted_model_names = sorted(available_models.keys(), key=lambda x: (x != 'baseline', x))
    for idx, model_name in enumerate(sorted_model_names, 1):
        print(f"   [{idx}] {model_name}")
    print("="*50)

    default_model = 'baseline' if 'baseline' in sorted_model_names else sorted_model_names[0]

    try:
        user_choice = input(f"Enter model to evaluate (number or name) [default: {default_model}]: ").strip().lower()
    except EOFError:
        user_choice = default_model
        print(default_model)

    # Resolve selection from user input
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
    print(f"--> Running evaluation for: {selected_name.upper()}\n")

    evaluator = Evaluator(num_workers=args.workers)
    evaluator.eval_model(selected_model_cls, model_name=selected_name, epochs=args.epochs)


if __name__ == "__main__":
    main()
