# Multiple Layers Test Case

Two layers defined.
* conv0, ends at 1.50, and runs for 400ms
* fc0, ends at 2.70, and runs for 800ms

conv0 should get two readings (at 1.20 and 1.40), resulting in mean of [1200*5000, 1300*5000] = 1250*5000 = 6250000

fc0 should get four readings (at 2.00, 2.20, 2.40 and 2.60), resulting in mean of [1600*5000, 1700*5000, 1800*5000, 1900*5000] = 1250*5000 = 8750000
