import random, time, os
from datetime import datetime
from math import sin, cos
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.widgets import Button
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition


class BallBeam():
    """ BallBeam

    Simple ball and beam simulation built to interface easily with OpenAI's
    gym environments.

    System dynamics
    ---------------
    dx/dt = v(t)
    dv/dt = -m*g*sin(theta(t))/((I + 1)*m)

    Parameters
    ----------
    time_step : time of one simulation step, float (s)

    beam_length : length of beam, float (units)

    max_angle : max of abs(angle), float (rads) 

    init_velocity : initial speed of ball, float (units/s)

    setpoint : fraction of beamwhere to balance the ball

    reward_scale : passesd to reward function to scale each reward component
    """

    def __init__(self, timestep=None, unit_conversion=None, beam_length=None, ball_radius=None, max_angle=None, max_ang_a=None, setpoint=None, init_velocity=None, sleep=None):
        self.g = 9.8
        self.dt = timestep                  
        self.ball_radius = ball_radius/unit_conversion  # cm
        self.L = beam_length/unit_conversion            # m
        self.beam_radius = self.L/2         
        self.setpoint = setpoint                        # fraction of beam where to balance ball
        self.point = self.setpoint*self.L               # actual balance point on beam (meters)
        self.I = 2/5*self.ball_radius**2                # solid ball inertia (omits mass)
        self.init_velocity = init_velocity/unit_conversion
        self.max_angle = max_angle
        self.max_ang_a = max_ang_a
        self.max_a = self.g*sin(max_angle)
        self.max_v = (self.init_velocity**2 + 10/7*self.g*self.L*sin(max_angle))**0.5
        self.reset()
        self.sleep = sleep
        self.human_rendered = self.machine_rendered = False
        self.ep = self.frame = 0

    def reset(self, setpoint=None, init_velocity=None):
        if setpoint is not None:
            self.setpoint = setpoint
            self.point = setpoint*self.L
        self.theta = 0
        self.ang_v = 0
        self.ang_a = 0
        self.x = random.uniform(-0.4,0.4)*self.L
        self.y = self.ball_radius# + 40 # ball is not actually rendered on top of the beam? (small radius ball is barely visible, as it's inside of the beam)
        self.v = self.init_velocity if init_velocity is None else init_velocity
        self.a = 0
        self.lim_x = [-self.beam_radius, self.beam_radius]
        self.lim_y = [0,0]
        self.t = 0
                      
    def update(self, action):
        """ 
        Update simulation with one time step

        Parameters
        ----------
        action : angle to which beam should be set, float (rad)
        """

        # simulation could be improved further by using voltage as input and a
        # motor simulation deciding theta

        # should be bounded already?, set action space mean to 0? and range from [-max, max]
        theta = max(-self.max_angle, min(self.max_angle, action.item())) 
        
        v_final = (theta - self.theta)/self.dt
        self.ang_a = (v_final - self.ang_v)/self.dt
        self.theta = theta
        self.ang_v = v_final

        """
        angular stuff is weird? trace shown below:
        
        timestep 1:
            theta = 0.2 rad
            self.theta = 0
            difference = 0.2
            dt = 0.05
            velocity = 4
                - sets self.ang_v = 4
            accel = (4-0)/0.05 = 80
         timestep 2:
            theta = 0.2 rad
            self.theta = 0.2
            differnce = 0
            velocity = 0
                - sets self.ang_v = 0
            accel = (0 - 4)/0.05 = -80
        timestep 3:
            theta = self.theta = 0.2
            difference = velocity = 0
            accel = 0
        
        """

        x = self.x
        v = self.v

        self.v += -self.g/(1 + 2/5)*sin(self.theta)*self.dt
        self.x += self.v*self.dt
        self.y = self.ball_radius/cos(self.theta) + self.x*sin(self.theta)
        
        self.v = (self.x - x)/self.dt
        self.a = (self.v - v)/self.dt
        
        self.lim_x = [-cos(self.theta)*self.beam_radius, cos(self.theta)*self.beam_radius]
        self.lim_y = [-sin(self.theta)*self.beam_radius, sin(self.theta)*self.beam_radius]
        
        self.t += self.dt

    def pause(self, event): 
        if self.sleep is not None:
            time.sleep(self.sleep)

    def screenshot(self, event):
        ss_dir = 'screenshots'
        if not os.path.exists(ss_dir):
            os.makedirs(ss_dir)
        fn = "ep={}_frame={}_tstep={}_time={}".format(self.ep, self.frame, round(self.t, 2), datetime.now().strftime("%H;%M;%S;%f")[:-3])
        self.btn_ss.set_visible(False)
        if self.sleep:
            self.btn_pause.set_visible(False)
        self.fig.savefig(f"{ss_dir}/{fn}.png", dpi=self.fig.dpi, bbox_inches='tight')
        self.btn_ss.set_visible(True)
        if self.sleep:
            self.btn_pause.set_visible(True)

    def _init_render(self, mode, draw_text=False):
        """ Initialize rendering """
        if mode == 'human':
            self.human_rendered = True
            plt.ion()
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            fig.canvas.manager.set_window_title('Ball & Beam')
            ax.set(xlim = (-2*self.beam_radius, 2*self.beam_radius), ylim = (-self.L/2, self.L/2))
            # ax.set_axis_off() # removes everything for blank background
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            
            # draw ball
            self.ball_plot = Circle((self.x, self.y), self.ball_radius)
            ax.add_patch(self.ball_plot)
            ax.patches[0].set_color('red')
            
            # draw beam
            ax.plot([-cos(self.theta)*self.beam_radius, cos(self.theta)*self.beam_radius],
                    [-sin(self.theta)*self.beam_radius, sin(self.theta)*self.beam_radius], lw=4, color='black')
            
            # draw pivot
            ax.plot(0.0, 0.0, '.', ms=15)

            # draw setpoint
            ax.add_patch(Polygon([
                [self.point*cos(self.theta), -0.01*self.L + self.point*sin(self.theta)],
                [self.point*cos(self.theta) - 0.015*self.L, -0.03*self.L + self.point*sin(self.theta)],
                [self.point*cos(self.theta) + 0.015*self.L, -0.03*self.L + self.point*sin(self.theta)]]))
            ax.patches[1].set_color('green')

            # draw screenshot button
            ss_ax = plt.axes([0, 0, 1, 1])
            ss_ax.set_axes_locator(InsetPosition(ax, [0.8, 0.8, 0.16, 0.1]))
            self.btn_ss = Button(ss_ax, "Screenshot")
            self.btn_ss.on_clicked(self.screenshot)

            # draw button
            if self.sleep:
                pause_ax = plt.axes([0, 0, 1, 1])
                pause_ax.set_axes_locator(InsetPosition(ax, [0.8, 0.68, 0.16, 0.1]))
                self.btn_pause = Button(pause_ax, "Pause")
                self.btn_pause.on_clicked(self.pause)

            # set blank components
            if draw_text:
                self.reward_text = ax.text(0.05, 0.95, "Rewards:\nr: 0.0\nv: 0.0\nθ: 0.0\nα: 0.0", transform=ax.transAxes, fontsize=10, va='top')

            self.fig = fig
            self.ax = ax
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        else:
            self.machine_rendered = True
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))

            # avoid drawing plot but still initialize
            _ = fig.canvas.manager.set_window_title('Ball & Beam')
            _ = ax.set(xlim = (-2*self.beam_radius, 2*self.beam_radius), ylim = (-self.L/2, self.L/2))
            
            # draw ball
            self.ball_plot = Circle((self.x, self.y), self.ball_radius)
            _ = ax.add_patch(self.ball_plot)
            _ = ax.patches[0].set_color('red')
            # draw beam
            _ = ax.plot([-cos(self.theta)*self.beam_radius, cos(self.theta)*self.beam_radius],
                        [-sin(self.theta)*self.beam_radius, sin(self.theta)*self.beam_radius], lw=4, color='black')
            _ = ax.plot(0.0, 0.0, '.', ms=20)
            _ = ax.add_patch(Polygon(
                [[self.point*cos(self.theta), -0.01*self.L + self.point*sin(self.theta)],
                 [self.point*cos(self.theta) - 0.015*self.L, - 0.03*self.L + self.point*sin(self.theta)],
                 [self.point*cos(self.theta) + 0.015*self.L, -0.03*self.L + self.point*sin(self.theta)]]))
            _ = ax.patches[1].set_color('green')

            self.machine_fig = fig
            self.machine_ax = ax

    def render(self, rc=None, button_info=None, mode='human'):
        """ 
        Render simulation at its current state

        Parameters
        ----------
        mode : rendering mode, str [human, machine]
        """
        if button_info is not None:
            self.ep, self.frame = button_info
        if (not self.human_rendered and mode == 'human') or (not self.machine_rendered and mode == 'machine'):
            self._init_render(mode, draw_text=rc is not None)
        elif mode == 'human': # i changed this to an elif, but forgot exactly why. i think it caused some sort of rare rendering bug
            # update ball
            self.ball_plot.set_center((self.x, self.y))
            
            # update beam
            self.ax.lines[0].set(xdata=self.lim_x, ydata=self.lim_y)
            
            # update setpoint
            self.ax.patches[1].set_xy([
                [self.point*cos(self.theta), -0.01*self.L + self.point*sin(self.theta)],
                [self.point*cos(self.theta) - 0.015*self.L, -0.03*self.L + self.point*sin(self.theta)],
                [self.point*cos(self.theta) + 0.015*self.L, -0.03*self.L + self.point*sin(self.theta)]])

            # write reward components
            if rc is not None:
                self.reward_text.set_text(f"Rewards:\nr: {round(rc['dist'], 3)}\nv: {round(rc['vel'], 3)}\nθ: {round(rc['ang'], 3)}\nα: {round(rc['ang_acc'], 3)}")
            
            # update figure
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        else:
            _ = self.ball_plot.set_center((self.x, self.y))
            _ = self.machine_ax.lines[0].set(xdata=self.lim_x, ydata=self.lim_y)
            _ = self.machine_ax.patches[1].set_xy([
                [self.point*cos(self.theta), -0.01*self.L + self.point*sin(self.theta)],
                [self.point*cos(self.theta) - 0.015*self.L, -0.03*self.L + self.point*sin(self.theta)],
                [self.point*cos(self.theta) + 0.015*self.L, -0.03*self.L + self.point*sin(self.theta)]])
            _ = self.machine_fig.canvas.draw()
            _ = self.machine_fig.canvas.flush_events()

    @property
    def on_beam(self):
        return self.lim_x[0] < self.x < self.lim_x[1]
