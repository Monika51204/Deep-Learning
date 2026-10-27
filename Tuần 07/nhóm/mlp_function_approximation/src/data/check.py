from src.data.load_data import load_data

X_train, y_train = load_data("train_pool", n_samples=400)
X_val, y_val = load_data("val")
X_test, y_test = load_data("test")

print(X_train.shape, y_train.shape)  # (400, 2), (400, 1)
print(X_val.shape, y_val.shape)      # (500, 2), (500, 1)
print(X_test.shape, y_test.shape)    # (100, 2), (100, 1)