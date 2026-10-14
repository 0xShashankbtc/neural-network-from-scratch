#!/usr/bin/env python3
"""
CLI Demonstration Entrypoint: Single-Neuron Neural Network from Scratch in NumPy.
"""

import sys
from src.model import SingleNeuronClassifier
from src.visualization import plot_dataset, plot_loss_curve, plot_decision_boundary


def run_pipeline() -> None:
    print("=" * 60)
    print("SINGLE-NEURON NEURAL NETWORK FROM SCRATCH IN NUMPY")
    print("=" * 60)

    # 1. Dataset definition
    dataset = [
        [3.0, 1.5, 1],
        [2.0, 1.0, 0],
        [4.0, 1.5, 1],
        [3.0, 1.0, 0],
        [3.5, 0.5, 1],
        [2.0, 0.5, 0],
        [5.5, 1.0, 1],
        [1.0, 1.0, 0],
    ]
    mystery_flower = [4.5, 1.0]

    print("\n1. Training Dataset (Petal Length, Petal Width, Class Label):")
    for row in dataset:
        label_str = "Red (1)" if row[2] == 1 else "Blue (0)"
        print(f"   Length: {row[0]:.1f}, Width: {row[1]:.1f} -> Label: {label_str}")

    print(f"\n   Unlabelled Mystery Query Sample: {mystery_flower}")

    # 2. Model Initialization & Training
    print("\n2. Initializing SingleNeuronClassifier and executing SGD optimization...")
    model = SingleNeuronClassifier(random_state=42)
    print(f"   Initial Random Weights: w1={model.w1:.4f}, w2={model.w2:.4f}, b={model.b:.4f}")

    iterations = 20000
    learning_rate = 0.1
    costs = model.fit(dataset, iterations=iterations, learning_rate=learning_rate)

    initial_loss = costs[0]
    final_loss = costs[-1]
    print(f"   Optimization complete ({iterations:,} iterations, alpha={learning_rate}).")
    print(f"   Initial Cumulative Loss: {initial_loss:.4f}")
    print(f"   Final Cumulative Loss:   {final_loss:.4f}")
    print(f"   Trained Parameters:     w1={model.w1:.4f}, w2={model.w2:.4f}, b={model.b:.4f}")

    # 3. Model Evaluation on Training Set
    print("\n3. Training Set Evaluation:")
    correct = 0
    for row in dataset:
        pred_prob = model.forward(row[0], row[1])
        pred_class = 1 if pred_prob >= 0.5 else 0
        actual = int(row[2])
        is_correct = pred_class == actual
        if is_correct:
            correct += 1
        status = "CORRECT" if is_correct else "MISMATCH"
        print(
            f"   Sample [{row[0]}, {row[1]}] -> Actual: {actual}, "
            f"Predicted Prob: {pred_prob:.4f} (Class: {pred_class}) [{status}]"
        )

    accuracy = (correct / len(dataset)) * 100.0
    print(f"\n   Training Set Accuracy: {accuracy:.1f}% ({correct}/{len(dataset)})")

    # 4. Out-of-Sample Inference
    print("\n4. Mystery Sample Classification:")
    q_length, q_width = mystery_flower
    query_prob = model.forward(q_length, q_width)
    query_class = model.predict(q_length, q_width)
    class_name = "Red Flower (Class 1)" if query_class == 1 else "Blue Flower (Class 0)"

    print(f"   Query Point: Petal Length = {q_length}, Petal Width = {q_width}")
    print(f"   Raw Activation Output (Sigmoid Probability): {query_prob:.4f}")
    print(f"   Predicted Classification:                   {class_name}")

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
