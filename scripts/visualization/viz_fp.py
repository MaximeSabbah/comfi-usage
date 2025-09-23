import numpy as np
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

viz = meshcat.Visualizer()
viz.open()  # or just print(viz.url()) and open it manually at 127.0.0.1:7002/static/

viz.delete()

def add_axes(viz, name="axes", length=1.0, lw=3):
    def line(p0, p1, color):
        pts = np.array([p0, p1]).T  # shape (3,2)
        return g.Line(
            g.PointsGeometry(pts),
            g.LineBasicMaterial(linewidth=lw, color=color)
        )
    viz[name]["x"].set_object(line([0,0,0],[length,0,0], 0xff0000))  # X red
    viz[name]["y"].set_object(line([0,0,0],[0,length,0], 0x00ff00))  # Y green
    viz[name]["z"].set_object(line([0,0,0],[0,0,length], 0x0000ff))  # Z blue

add_axes(viz, length=1.0)

# Put a cylinder to test default axis alignment
cyl = g.Cylinder(1.0, 0.05)  # height=1 (along default axis), radius=0.05
viz["cyl"].set_object(cyl, g.MeshLambertMaterial(color=0xffaa00))
viz["cyl"].set_transform(tf.translation_matrix([0, 0.5, 0]))  # shift so it runs from (0,0,0) to (0,1,0)
