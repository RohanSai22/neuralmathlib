import streamlit as st
import numpy as np

# Define classes for Neural Math Library
class LinearAlgebra:
    @staticmethod
    def dot_product(A, B):
        return np.dot(A, B)

    @staticmethod
    def element_wise_addition(A, B):
        return np.add(A, B)

    @staticmethod
    def element_wise_multiplication(A, B):
        return np.multiply(A, B)

    @staticmethod
    def matrix_transpose(A):
        return np.transpose(A)

    @staticmethod
    def matrix_inverse(A):
        return np.linalg.inv(A)

    @staticmethod
    def determinant(A):
        return np.linalg.det(A)

class ActivationFunctions:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return ActivationFunctions.sigmoid(x) * (1 - ActivationFunctions.sigmoid(x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.square(ActivationFunctions.tanh(x))

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class LossFunctions:
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

class Optimizers:
    @staticmethod
    def gradient_descent(weights, gradients, learning_rate=0.01):
        return weights - learning_rate * gradients

    @staticmethod
    def adam_optimizer(weights, gradients, m, v, t, beta1=0.9, beta2=0.999, epsilon=1e-8, learning_rate=0.001):
        m = beta1 * m + (1 - beta1) * gradients
        v = beta2 * v + (1 - beta2) * np.square(gradients)
        m_hat = m / (1 - np.power(beta1, t))
        v_hat = v / (1 - np.power(beta2, t))
        weights = weights - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        return weights, m, v

class Utilities:
    @staticmethod
    def initialize_weights(shape, method='xavier'):
        if method == 'xavier':
            return np.random.randn(*shape) / np.sqrt(shape[0] / 2)
        elif method == 'he':
            return np.random.randn(*shape) * np.sqrt(2. / shape[0])
        else:
            return np.random.randn(*shape)

    @staticmethod
    def normalize_batch(X):
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    @staticmethod
    def softmax_cross_entropy_loss(y_true, y_pred):
        return LossFunctions.cross_entropy_loss(y_true, ActivationFunctions.softmax(y_pred))

# Streamlit App UI
def run_neural_math_app():
    st.title("Neural Math Library - Deep Learning")

    st.sidebar.header("Test Operations")
    option = st.sidebar.selectbox(
        "Choose an operation to test:",
        ("Matrix Operations", "Activation Functions", "Loss Functions", "Optimization Algorithms")
    )

    if option == "Matrix Operations":
        matrix_operations()

    elif option == "Activation Functions":
        activation_functions()

    elif option == "Loss Functions":
        loss_functions()

    elif option == "Optimization Algorithms":
        optimization_algorithms()

def matrix_operations():
    st.header("Matrix Operations")
    
    # Matrix Inputs
    rows_A = st.number_input("Rows in Matrix A", min_value=1, max_value=5, value=2)
    cols_A = st.number_input("Columns in Matrix A", min_value=1, max_value=5, value=2)
    rows_B = st.number_input("Rows in Matrix B", min_value=1, max_value=5, value=2)
    cols_B = st.number_input("Columns in Matrix B", min_value=1, max_value=5, value=2)

    A = np.random.rand(rows_A, cols_A)
    B = np.random.rand(rows_B, cols_B)
    
    # Display Matrices
    st.write("Matrix A:")
    st.write(A)
    st.write("Matrix B:")
    st.write(B)
    
    # Operations
    if rows_A == cols_B:
        st.write("Dot Product of A and B:")
        st.write(LinearAlgebra.dot_product(A, B))

    st.write("Element-wise Addition of A and B:")
    try:
        st.write(LinearAlgebra.element_wise_addition(A, B))
    except ValueError as e:
        st.write("Matrices must be of the same shape!")

    st.write("Element-wise Multiplication of A and B:")
    try:
        st.write(LinearAlgebra.element_wise_multiplication(A, B))
    except ValueError as e:
        st.write("Matrices must be of the same shape!")

def activation_functions():
    st.header("Activation Functions")
    
    # Input
    x = st.text_area("Enter a list of values separated by commas (e.g., [1, -1, 0, 2])", "[1, -1, 0, 2]")
    x = np.array(eval(x))
    
    function_choice = st.selectbox("Choose an activation function", ("Sigmoid", "ReLU", "Tanh", "Softmax"))
    
    if function_choice == "Sigmoid":
        st.write("Sigmoid Function:")
        st.write(ActivationFunctions.sigmoid(x))

    elif function_choice == "ReLU":
        st.write("ReLU Function:")
        st.write(ActivationFunctions.relu(x))

    elif function_choice == "Tanh":
        st.write("Tanh Function:")
        st.write(ActivationFunctions.tanh(x))

    elif function_choice == "Softmax":
        st.write("Softmax Function:")
        st.write(ActivationFunctions.softmax(x))

def loss_functions():
    st.header("Loss Functions")

    # Input
    y_true = st.text_area("Enter true labels (e.g., [[0, 1], [1, 0]])", "[[0, 1], [1, 0]]")
    y_true = np.array(eval(y_true))
    
    y_pred = st.text_area("Enter predicted labels (e.g., [[0.2, 0.8], [0.6, 0.4]])", "[[0.2, 0.8], [0.6, 0.4]]")
    y_pred = np.array(eval(y_pred))

    loss_choice = st.selectbox("Choose a loss function", ("Mean Squared Error", "Cross Entropy Loss"))

    if loss_choice == "Mean Squared Error":
        st.write("Mean Squared Error Loss:")
        st.write(LossFunctions.mean_squared_error(y_true, y_pred))

    elif loss_choice == "Cross Entropy Loss":
        st.write("Cross Entropy Loss:")
        st.write(LossFunctions.cross_entropy_loss(y_true, y_pred))

def optimization_algorithms():
    st.header("Optimization Algorithms")

    # Initial Weights and Gradients Input
    weights = st.text_area("Enter initial weights (e.g., [[0.1, 0.2], [0.3, 0.4]])", "[[0.1, 0.2], [0.3, 0.4]]")
    weights = np.array(eval(weights))
    
    gradients = st.text_area("Enter gradients (e.g., [[0.05, -0.02], [-0.03, 0.07]])", "[[0.05, -0.02], [-0.03, 0.07]]")
    gradients = np.array(eval(gradients))

    optimizer_choice = st.selectbox("Choose an optimizer", ("Gradient Descent", "Adam Optimizer"))

    if optimizer_choice == "Gradient Descent":
        learning_rate = st.number_input("Enter learning rate", min_value=0.001, max_value=1.0, value=0.01)
        updated_weights = Optimizers.gradient_descent(weights, gradients, learning_rate)
        st.write("Updated Weights after Gradient Descent:")
        st.write(updated_weights)

    elif optimizer_choice == "Adam Optimizer":
        m = np.zeros_like(weights)
        v = np.zeros_like(weights)
        t = 1
        updated_weights, m, v = Optimizers.adam_optimizer(weights, gradients, m, v, t)
        st.write("Updated Weights after Adam Optimizer:")
        st.write(updated_weights)

if __name__ == "__main__":
    run_neural_math_app()
