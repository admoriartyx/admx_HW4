# This is the .py file for Question 2 of Problem Set 4

# Part a

import numpy as np
import matplotlib.pyplot as plt

def julia_set(c, xmin, xmax, ymin, ymax, xpoints, ypoints, max_iter):
    x = np.linspace(xmin, xmax, xpoints)
    y = np.linspace(ymin, ymax, ypoints)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    img = np.zeros(Z.shape, dtype=int)

    for i in range(max_iter):
        Z = Z**2 + c 
        mask = np.abs(Z) < 1000
        img += mask
    return x, y, img

c = -0.7 + 0.356j
xmin = -1.5
xmax = 1.5
ymin = -1
ymax = 1
xpoints = 800
ypoints = 800
max_iter = 256
x, y, img = julia_set(c, xmin, xmax, ymin, ymax, xpoints, ypoints, max_iter)

plt.figure(figsize=(8, 8))
plt.imshow(img, extent=(xmin, xmax, ymin, ymax), cmap="hot")
plt.title("Julia Set for $f(z) = z^2 + (-0.7 + 0.356i)$")
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.show()
plt.savefig('Q2_part_a.png')

# Part b

from scipy.spatial import ConvexHull

def julia_set_pixel(z0, c, max_iter, r):
    z = z0
    for i in range(max_iter):
        if abs(z) > r:
            return i
        z = z**2 + c
    return max_iter

julia_set_pixel = np.vectorize(julia_set_pixel)

re_min = -1.5
re_max = 1.5
im_min = -1
im_max = 1

re_width = 800
im_width = 800
re, im = np.meshgrid(np.linspace(re_min, re_max, re_width), np.linspace(im_min, im_max, im_width))
z0 = re + 1j * im
c = -0.7 + 0.356j
max_iter = 256
r = 2
pixels = julia_set_pixel(z0, c, max_iter, r)

def hull_plotter(points):
    xs, ys = zip(*points)
    plt.plot(xs + xs[:1], ys + ys[:1], 'r-', linewidth=2, label='Convex Hull')

def hull_area(polygon):
    xs, ys = np.array(polygon).T
    return 0.5 * np.abs(sum(np.roll(xs, 1) * ys - xs * np.roll(ys, 1)))

julia_threshold = max_iter / 2
pixels_bool = pixels > julia_threshold
points = list(zip(re[pixels_bool], im[pixels_bool]))

if points:
    hull = ConvexHull(points)
    hull_points = [points[v] for v in hull.vertices]
    area = hull_area(hull_points)
    print(f'Area of the convex hull: {area:.2f}')
    hull_plotter(hull_points)
    plt.imshow(pixels, extent=(re_min, re_max, im_min, im_max), cmap='inferno', origin='lower')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.show()
else:
    print("No points found for convex hull calculation.")

plt.savefig('part_b_test2.png')

# Part c
