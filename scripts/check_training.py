import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cli
from dataset_loader import DATASET_CLASSES, TrafficSignDataset
from models.aco.model import ACOModel, AntColonyFeatureOptimizer
from models.adabn.model import AdaBN
from models.baseline.model import Baseline
from noise import CorruptedDataset


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cpu')
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        Image.new('RGB', (49, 49), (80, 100, 120)).save(root / 'sign.png')
        (root / 'Test.csv').write_text('Path,ClassId\nsign.png,0\nsign.png,0\n', encoding='utf-8')
        dataset = TrafficSignDataset(split='test', root_dir=str(root))
        first_image, first_label = dataset.get_pil_image(0)
        for attempt in range(3):
            cached_image, cached_label = dataset.get_pil_image(0)
            assert cached_image is first_image
            assert cached_label == first_label == 0
        corrupted = CorruptedDataset(dataset, noise_type='brightness')
        loader = DataLoader(corrupted, batch_size=1, num_workers=0)
        for attempt in range(3):
            images, labels = next(iter(loader))
            assert images.shape == (1, 3, 49, 49)
            assert labels.tolist() == [0]

        optimize = AntColonyFeatureOptimizer.optimize
        search_calls = []

        def checked_optimize(optimizer, feature_extractor, classifier, sample_loader, device):
            model = feature_extractor.__self__
            assert not model.training
            assert not model.drop1.training and not model.drop2.training
            search_calls.append(model.num_classes)
            return optimize(optimizer, feature_extractor, classifier, sample_loader, device)

        for dataset_name, class_count in DATASET_CLASSES.items():
            training_data = TensorDataset(torch.rand(2, 3, 49, 49), torch.tensor([0, class_count - 1]))
            training_data.dataset_name = dataset_name
            for model_class in (Baseline, AdaBN, ACOModel):
                model = model_class(num_classes=class_count)
                assert not any(isinstance(layer, torch.nn.GELU) for layer in model.modules())
                assert sum(isinstance(layer, torch.nn.ReLU) for layer in model.modules()) == 4
                destination = root / dataset_name / model_class.__name__
                options = {'num_ants': 2, 'aco_iterations': 1} if model_class is ACOModel else {}
                with patch.object(AntColonyFeatureOptimizer, 'optimize', checked_optimize):
                    metrics = model.train(training_data, epochs=2, batch_size=2, device=device,
                                          save_dir=str(destination), **options)
                assert math.isfinite(metrics['loss'])
                assert model.training
                predictions, labels = model.test(training_data, batch_size=2, device=device)
                assert len(predictions) == len(labels) == 2
                checkpoint = next(destination.glob('*.pth'))
                restored = model_class(num_classes=class_count)
                assert restored.load_weights(str(checkpoint))
                with torch.no_grad():
                    assert restored.eval()(training_data.tensors[0]).shape == (2, class_count)
                wrong_head = model_class(num_classes=class_count + 1)
                original_weight = next(wrong_head.parameters()).detach().clone()
                assert not wrong_head.load_weights(str(checkpoint))
                assert torch.equal(next(wrong_head.parameters()), original_weight)
                if model_class is ACOModel:
                    assert torch.equal(restored.pheromone_gate, model.pheromone_gate)
                    assert torch.all(model.pheromone_gate >= 0.5)
                    assert torch.all(model.pheromone_gate <= 1.5)
                    assert not torch.equal(model.pheromone_gate, torch.ones_like(model.pheromone_gate))
                if model_class is AdaBN:
                    metadata = json.loads((destination / 'model_metadata.json').read_text())
                    assert metadata['dataset'] == dataset_name
                    parameters = [parameter.detach().clone() for parameter in model.parameters()]
                    model.adapt_bn(corrupted, batch_size=2, device=device, verbose=False)
                    assert len(model.test(corrupted, batch_size=2, device=device)[0]) == 2
                    assert all(torch.equal(before, after) for before, after in zip(parameters, model.parameters()))
                    assert not model.training
        assert search_calls == list(DATASET_CLASSES.values())

        failed_model = ACOModel()
        training_data = TensorDataset(torch.rand(2, 3, 49, 49), torch.tensor([0, 1]))
        with patch.object(AntColonyFeatureOptimizer, 'optimize', side_effect=RuntimeError('search failed')):
            try:
                failed_model.train(training_data, epochs=2, batch_size=2, device=device, save_dir=None)
            except RuntimeError as error:
                assert str(error) == 'search failed'
            else:
                raise AssertionError('Expected search failure')
        assert failed_model.training and failed_model.drop1.training and failed_model.drop2.training

        for retrain in (False, True):
            arguments = ['cli.py', '--model', 'baseline', '--dataset', 'btsd']
            if retrain:
                arguments.append('--retrain')
            with patch.object(sys, 'argv', arguments), patch.object(cli, 'Evaluator') as evaluator:
                cli.main()
                assert evaluator.return_value.eval_model.call_args.kwargs['load_if_exists'] is not retrain
        with patch.object(sys, 'argv', ['cli.py', '--retrain', '--weights', 'checkpoint.pth']):
            try:
                cli.main()
            except SystemExit as error:
                assert error.code == 2
            else:
                raise AssertionError('Conflicting checkpoint options were accepted')
    print('PASS: PIL reuse, ReLU, all model/class-count training paths, ACO modes, checkpoints, metadata, and retrain CLI.')


if __name__ == '__main__':
    main()
