import matplotlib.pyplot as plt

# Example data
speed = [44.72, 48.7, 45, 53.76, 83, 47.09, 49.79, 60.86]  # Model sizes
variation = ['resnet18', 'resnet18-bottleneck', 'resnet18-KD', 'resnet50(baseline)', 'swinT(baseline)', 'poolformer-resnet50', 'lidar-small', 'lidar-large']  # Detection heads
mAP = [0.64, 0.658, 0.65, 0.679, 0.685, 0.606, 0.654, 0.671]  # Example accuracy scores

colors = ['red', 'red', 'red','green', 'blue', 'orange', 'magenta', 'magenta']
shapes = ['o', 'x', 's', 'o', 'o', 'o', 'o', 'x']

# # Plotting the results
# plt.figure(figsize=(6, 10))
# plt.scatter(speed, mAP, s=100, c='blue', label='Model Size vs. Accuracy')
# for i, txt in enumerate(variation):
#     plt.annotate(txt, (speed[i], mAP[i]), textcoords="offset points", xytext=(0,10), ha='center')
# plt.xlabel('Speed (ms)')
# plt.ylabel('mAP')
# plt.title('Speed vs. mAP for BEVFusion variation')
# plt.grid(False)
# plt.legend()

# plt.savefig('speed_vs_accuracy.png', bbox_inches='tight')  # Specify the filename and adjust the bounding box
# plt.show()


fig, ax = plt.subplots(figsize=(6, 6))
for i , (vel, score, color, shape) in enumerate(zip(speed, mAP, colors, shapes)) :
    ax.scatter(vel, score, s=100,c=color, marker=shape, label=variation[i])
ax.set_xlabel('Speed')
ax.set_ylabel('mAP')
ax.set_title('Speed vs. mAP for variation of BEVFusion (fp16)')
ax.grid(True)
ax.legend()

plt.savefig('speed_vs_accuracy_fp16.png', bbox_inches='tight')  # Specify the filename and adjust the bounding box
plt.show()
