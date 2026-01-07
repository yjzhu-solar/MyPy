import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Wedge
from matplotlib.collections import PatchCollection
from datetime import datetime, timedelta
import matplotlib.animation as animation

class AnalogClock:
    """
    A matplotlib-based analog clock using patches that can be animated.
    
    Parameters
    ----------
    center : tuple
        (x, y) coordinates for clock center
    radius : float
        Radius of the clock face
    facecolor : str
        Color of the clock face (default: 'orange')
    edgecolor : str
        Color of clock border (default: 'white')
    tick_color : str
        Color of hour tick marks (default: 'white')
    hand_color : str
        Color of clock hands (default: 'white')
    alpha : float
        Transparency of all clock elements (default: 1.0)
    zorder : int
        Base z-order for clock elements (default: 1)
    """
    
    def __init__(self, center=(0, 0), radius=1.0, facecolor='#FF6B35', 
                 edgecolor='white', tick_color='white', hand_color='white',
                 alpha=1.0, zorder=1):
        self.center = np.array(center)
        self.radius = radius
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.tick_color = tick_color
        self.hand_color = hand_color
        self.alpha = alpha
        self.base_zorder = zorder
        
        # Store patches
        self.face_patch = None
        self.tick_patches = []
        self.hour_hand = None
        self.minute_hand = None
        self.all_patches = []
        
    def _create_face(self):
        """Create the circular clock face"""
        self.face_patch = Circle(self.center, self.radius, 
                                facecolor=self.facecolor, 
                                edgecolor=self.edgecolor, 
                                linewidth=2, 
                                alpha=self.alpha,
                                zorder=self.base_zorder)
        return self.face_patch
    
    def _create_tick_marks(self):
        """Create hour tick marks (triangles and diamonds)"""
        self.tick_patches = []
        
        for hour in range(12):
            angle = np.radians(90 - hour * 30)  # Start at 12 o'clock
            dangle = np.radians(15)
            
            # Distance from center for tick marks
            tick_distance = 0.9 * self.radius

            # Create different shapes for different hours
            if hour % 3 == 0:  # 12, 3, 6, 9 - larger triangles
                size = 0.15 * self.radius
                tick_x = self.center[0] + (tick_distance - size*2/3*np.cos(dangle)) * np.cos(angle)
                tick_y = self.center[1] + (tick_distance - size*2/3*np.cos(dangle)) * np.sin(angle)
                vertices = [
                    (tick_x, tick_y),
                    (tick_x + size*np.cos(angle - dangle), tick_y + size*np.sin(angle - dangle)),
                    (tick_x + size*np.cos(angle + dangle), tick_y + size*np.sin(angle + dangle))
                ]

                patch = Polygon(vertices, facecolor=self.tick_color, 
                        edgecolor='none', 
                        alpha=self.alpha,
                        zorder=self.base_zorder + 1)

            else:  # Other hours - small dots
                size = 0.025 * self.radius
                tick_x = self.center[0] + tick_distance * np.cos(angle)
                tick_y = self.center[1] + tick_distance * np.sin(angle)

                patch = Circle((tick_x, tick_y), size, 
                            facecolor=self.tick_color, 
                            edgecolor='none', 
                            alpha=self.alpha,
                            zorder=self.base_zorder + 1)
            

            self.tick_patches.append(patch)
        
        return self.tick_patches
    
    def _create_hand(self, length_top, length_bottom, width, angle):
        """
        Create a clock hand as a polygon
        
        Parameters
        ----------
        length : float
            Length of the hand relative to radius
        width : float
            Width of the hand base relative to radius
        angle : float
            Angle in degrees (0 = 12 o'clock, clockwise)
        """
        angle_rad = np.radians(90 - angle)  # Convert to standard math angle
        
        # Hand vertices (arrow shape)
        tip_x = self.center[0] + (length_top + length_bottom) * self.radius * np.cos(angle_rad)
        tip_y = self.center[1] + (length_top + length_bottom) * self.radius * np.sin(angle_rad)
        
        base_offset = width * self.radius / 2
        base_x1 = self.center[0] + length_bottom * self.radius * np.cos(angle_rad) - base_offset * np.sin(angle_rad)
        base_y1 = self.center[1] + length_bottom * self.radius * np.sin(angle_rad) + base_offset * np.cos(angle_rad)
        base_x2 = self.center[0] + length_bottom * self.radius * np.cos(angle_rad) + base_offset * np.sin(angle_rad)
        base_y2 = self.center[1] + length_bottom * self.radius * np.sin(angle_rad) - base_offset * np.cos(angle_rad)
        
        vertices = [(tip_x, tip_y), (base_x1, base_y1), (self.center[0], self.center[1]), (base_x2, base_y2)]
        
        hand = Polygon(vertices, facecolor=self.hand_color, 
                      edgecolor='none', 
                      alpha=self.alpha,
                      zorder=self.base_zorder + 2)
        return hand
    
    def set_time(self, dt):
        """
        Set clock hands based on datetime object
        
        Parameters
        ----------
        dt : datetime
            Datetime object to display
        """
        hour = dt.hour % 12
        minute = dt.minute
        second = dt.second
        
        # Calculate angles (degrees from 12 o'clock, clockwise)
        # Minute hand: 6 degrees per minute
        minute_angle = minute * 6 + second * 0.1
        
        # Hour hand: 30 degrees per hour + 0.5 degrees per minute
        hour_angle = hour * 30 + minute * 0.5
        
        # Create hands
        self.hour_hand = self._create_hand(length_bottom=0.25, length_top=0.25, width=0.10, angle=hour_angle)
        self.minute_hand = self._create_hand(length_bottom=0.28, length_top=0.45, width=0.08, angle=minute_angle)
        
        return self.hour_hand, self.minute_hand
    
    def add_to_axes(self, ax, dt=None):
        """
        Add all clock patches to matplotlib axes
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to add patches to
        dt : datetime, optional
            Initial time to display (default: current time)
        """
        if dt is None:
            dt = datetime.now()
        
        # Clear previous patches list
        self.all_patches = []
        
        # Add face
        self.face_patch = self._create_face()
        ax.add_patch(self.face_patch)
        self.all_patches.append(self.face_patch)
        
        # Add tick marks
        for patch in self._create_tick_marks():
            ax.add_patch(patch)
            self.all_patches.append(patch)
        
        # Add hands
        self.set_time(dt)
        ax.add_patch(self.hour_hand)
        ax.add_patch(self.minute_hand)
        self.all_patches.extend([self.hour_hand, self.minute_hand])
        
        # Set equal aspect and limits
        margin = 0.1 * self.radius
        ax.set_xlim(self.center[0] - self.radius - margin, 
                   self.center[0] + self.radius + margin)
        ax.set_ylim(self.center[1] - self.radius - margin, 
                   self.center[1] + self.radius + margin)
        ax.set_aspect('equal')
        
        return ax
    
    def update_time(self, dt):
        """
        Update clock hands for animation
        
        Parameters
        ----------
        dt : datetime
            New time to display
            
        Returns
        -------
        list
            Updated patches
        """
        hour = dt.hour % 12
        minute = dt.minute
        second = dt.second
        
        minute_angle = minute * 6 + second * 0.1
        hour_angle = hour * 30 + minute * 0.5
        
        # Update minute hand
        minute_angle_rad = np.radians(90 - minute_angle)
        length_bottom = 0.28
        length_top = 0.45
        minute_width = 0.08

        tip_x = self.center[0] + (length_top + length_bottom) * self.radius * np.cos(minute_angle_rad)
        tip_y = self.center[1] + (length_top + length_bottom) * self.radius * np.sin(minute_angle_rad)
        
        base_offset = minute_width * self.radius / 2
        base_x1 = self.center[0] + length_bottom * self.radius * np.cos(minute_angle_rad) - base_offset * np.sin(minute_angle_rad)
        base_y1 = self.center[1] + length_bottom * self.radius * np.sin(minute_angle_rad) + base_offset * np.cos(minute_angle_rad)
        base_x2 = self.center[0] + length_bottom * self.radius * np.cos(minute_angle_rad) + base_offset * np.sin(minute_angle_rad)
        base_y2 = self.center[1] + length_bottom * self.radius * np.sin(minute_angle_rad) - base_offset * np.cos(minute_angle_rad)

        self.minute_hand.set_xy([(tip_x, tip_y), (base_x1, base_y1), (self.center[0], self.center[1]), (base_x2, base_y2)])
        
        # Update hour hand
        hour_angle_rad = np.radians(90 - hour_angle)
        length_bottom = 0.25
        length_top = 0.25
        hour_width = 0.10

        tip_x = self.center[0] + (length_top + length_bottom) * self.radius * np.cos(hour_angle_rad)
        tip_y = self.center[1] + (length_top + length_bottom) * self.radius * np.sin(hour_angle_rad)
        
        base_offset = hour_width * self.radius / 2
        base_x1 = self.center[0] + length_bottom * self.radius * np.cos(hour_angle_rad) - base_offset * np.sin(hour_angle_rad)
        base_y1 = self.center[1] + length_bottom * self.radius * np.sin(hour_angle_rad) + base_offset * np.cos(hour_angle_rad)
        base_x2 = self.center[0] + length_bottom * self.radius * np.cos(hour_angle_rad) + base_offset * np.sin(hour_angle_rad)
        base_y2 = self.center[1] + length_bottom * self.radius * np.sin(hour_angle_rad) - base_offset * np.cos(hour_angle_rad)

        self.hour_hand.set_xy([(tip_x, tip_y), (base_x1, base_y1), (self.center[0], self.center[1]), (base_x2, base_y2)])
        
        return [self.hour_hand, self.minute_hand]
    
    def set_alpha(self, alpha):
        """
        Set transparency for all clock elements
        
        Parameters
        ----------
        alpha : float
            Transparency value between 0 (transparent) and 1 (opaque)
        """
        self.alpha = alpha
        
        # Update all patches
        if self.face_patch is not None:
            self.face_patch.set_alpha(alpha)
        
        for patch in self.tick_patches:
            patch.set_alpha(alpha)
        
        if self.hour_hand is not None:
            self.hour_hand.set_alpha(alpha)
        
        if self.minute_hand is not None:
            self.minute_hand.set_alpha(alpha)
    
    def set_zorder(self, zorder):
        """
        Set z-order for all clock elements
        
        Parameters
        ----------
        zorder : int
            Base z-order value. Components will be stacked relative to this:
            - Face: zorder
            - Ticks: zorder + 1
            - Hands: zorder + 2
            - Center: zorder + 3
        """
        self.base_zorder = zorder
        
        # Update all patches
        if self.face_patch is not None:
            self.face_patch.set_zorder(zorder)
        
        for patch in self.tick_patches:
            patch.set_zorder(zorder + 1)
        
        if self.hour_hand is not None:
            self.hour_hand.set_zorder(zorder + 2)
        
        if self.minute_hand is not None:
            self.minute_hand.set_zorder(zorder + 2)
    
    def set_alpha_components(self, face=None, ticks=None, hands=None, center=None):
        """
        Set transparency for individual clock components
        
        Parameters
        ----------
        face : float, optional
            Alpha for clock face
        ticks : float, optional
            Alpha for tick marks
        hands : float, optional
            Alpha for hour and minute hands
        center : float, optional
            Alpha for center circle
        """
        if face is not None and self.face_patch is not None:
            self.face_patch.set_alpha(face)
        
        if ticks is not None:
            for patch in self.tick_patches:
                patch.set_alpha(ticks)
        
        if hands is not None:
            if self.hour_hand is not None:
                self.hour_hand.set_alpha(hands)
            if self.minute_hand is not None:
                self.minute_hand.set_alpha(hands)
    
    def get_all_patches(self):
        """
        Get list of all patches for external manipulation
        
        Returns
        -------
        list
            All patches in the clock
        """
        return self.all_patches
    
    def remove(self):
        """
        Remove all clock patches from the axes
        
        This cleanly removes all patches that were added to the axes,
        allowing the clock to be hidden or deleted from the figure.
        """
        for patch in self.all_patches:
            if patch is not None:
                patch.remove()
        
        # Clear internal references
        self.face_patch = None
        self.tick_patches = []
        self.hour_hand = None
        self.minute_hand = None
        self.all_patches = []