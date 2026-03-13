import torch
import random
from utils import generate

def test_robustness(model, test_data, num_samples=100, noise_level=0.0):
    model.eval()
    correct = 0
    # ensure we don't sample more than available
    n_test = min(num_samples, len(test_data))
    samples = random.sample(test_data, n_test)

    for idx, (input_str, target_str) in enumerate(samples):
        # Optional noise injection
        noisy_input = ''.join(random.choice('0123456789+= ') if random.random() < noise_level else c for c in input_str)
        generated = generate(model, noisy_input)

        if generated.strip() == target_str.strip():
            correct += 1

        if idx < 5:
            print(f"Example {idx}: prompt='{noisy_input}', generated='{generated}', expected='{target_str}'")

    accuracy = correct / n_test
    print(f"Accuracy on test set: {accuracy:.2%}")
    return accuracy
