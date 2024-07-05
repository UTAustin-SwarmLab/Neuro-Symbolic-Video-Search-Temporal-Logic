import pickle

import matplotlib.pyplot as plt

with open(
    "/opt/Neuro-Symbolic-Video-Frame-Search/experiments/data/res_clipFprop1_v2.pkl",
    "rb",
) as file:
    res_dict = pickle.load(file)

category = "accuracy"
data = []
for key, value in res_dict.items():
    data.append(value[category])

# np.random.seed(0)
# data = np.random.randn(50, 5) # Creating 5 sets of random data

# Creating the box plot
fig, ax = plt.subplots()

# Plotting the boxplot
ax.boxplot(
    data,
    notch=True,  # notch shape
    vert=True,  # vertical box alignment
    patch_artist=True,  # fill with color
    meanline=True,  # show mean line
    showmeans=True,  # show mean values
    showfliers=False,  # this will hide the outliers
)

# Adding error bars. Here, I'm adding error bars for the mean values for demonstration
# You can calculate other statistics like standard error and plot error bars based on that
# for i in range(5):
#     x = np.ones(data[:, i].shape) * (i+1)
#     ax.errorbar(x, data[:, i], yerr=np.std(data[:, i]), fmt='none', color='red')

# Labelling the x-axis with the categories
ax.set_xticklabels(res_dict.keys())

ax.set_xlabel("Manual Threshold")  # Add x axis label
ax.set_ylabel("Accuracy")  # Add y axis label


# Display the plot
plt.savefig("/opt/Neuro-Symbolic-Video-Frame-Search/experiments/boxplot.png")
