from network import Neural_Network
from mnist_loader import prepare_data

def main():
    network_size = [784, 20, 20, 10]
    neural_network = Neural_Network(network_size)
    training_data, test_data = prepare_data()
    neural_network.train(training_data, 20, 100, 0.1)
    neural_network.test(test_data)

if __name__ == "__main__":
    main()