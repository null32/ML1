import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

# cyan
X1, Y1 = make_gaussian_quantiles(
    n_features = 2, n_classes = 2,
    n_samples = 200,
    cov = 1,
    mean = (2, -2),
    random_state = 1
)
# magenta
X2, Y2 = make_gaussian_quantiles(
    n_features = 2, n_classes=2,
    n_samples = 300,
    cov = 2.5,
    mean = (5, 3),
    random_state = 1
)
# dataset
X = np.concatenate((X1, X2))
Y = np.concatenate((Y1, 1 - Y2))

bdt = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    algorithm = "SAMME",
    n_estimators = 200
)

bdt.fit(X, Y)

plot_colors = ["#40FFFF", "#FF40FF"]
plot_step = 0.02
class_names = "AB"
cmap = 'cool'
# Plot size
plt.figure(figsize = (10, 5))

# Plot boundaries
plt.subplot(1, 2, 1)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, plot_step),
    np.arange(y_min, y_max, plot_step)
)

# Predict surface
Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap = cmap)
plt.axis("tight")

# Plot data
for i in range(2):
    cl_name = class_names[i]
    pl_color = plot_colors[i]
    indexes = np.where(Y == i)
    plt.scatter(
        X[indexes, 0], X[indexes, 1],
        c = pl_color, cmap = cmap,
        s = 20, edgecolors = 'k',
        label = f'Class {cl_name}')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc = 'upper right')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Boundary')

# Plot the two-class decision scores
twoclass_output = bdt.decision_function(X)
plot_range = (twoclass_output.min(), twoclass_output.max())
plt.subplot(1, 2, 2)
for i in range(2):
    cl_name = class_names[i]
    pl_color = plot_colors[i]
    plt.hist(
        twoclass_output[Y == i],
        bins = 10, range = plot_range,
        facecolor = pl_color,
        alpha = 0.5, edgecolor = 'k',
        label = f'Class {cl_name}'
    )

x1, x2, Y1, Y2 = plt.axis()
plt.axis((x1, x2, Y1, Y2 * 1.2))
plt.legend(loc = 'upper right')
plt.ylabel('Samples')
plt.xlabel('Score')
plt.title('Decision Scores')

plt.tight_layout()
plt.subplots_adjust(wspace=0.35)
plt.show()