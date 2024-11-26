from manim import *


class CreateCircle(Scene):
    def construct(self):
        self.set_camera_orientation(phi=90 * DEGREES)  # set the camera to a side view
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set the color and transparency
        self.play(Create(circle))  # show the circle on screene

class CreateFormula(Scene):
    def construct(self):
        formula = MathTex(r"\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}")
        self.play(Create(formula))
        
        self.wait(2)
        
        self.play(Uncreate(formula))
        
        self.wait(1)
        
        #end of the sceneII
        
class Attractor(Scene):
    
    def construct(self):
        func = lambda pos: ((pos[0]*UR+pos[1]*LEFT) - pos)  
        mob= StreamLines(func,x_range=[-5,5,1], y_range=[-5,5,1],stroke_width=3)  

        # animate smoothly rhe creation of the streamlines
        self.play(Create(mob), run_time=5)
        
        
    
    