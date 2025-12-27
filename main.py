import eval
import train
import matplotlib.pyplot as plt

def main():

    batch_size = 64
    learning_rate = 0.001
    epochs = 5

    model, train_losses, test_losses = train.train(batch_size, learning_rate, epochs)

    acc = eval.evaluate(model, batch_size)

    print(f"Accuracy = {acc:.4f}")

    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()