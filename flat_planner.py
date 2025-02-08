CONFIG = dict(
room_edge_thickness = 4,
padding = 50,
dmarker_width = 3,
dmarker_offset = -20,
highlight_vertices = False,
debug_plot = False,
outlet_length = 20,
)

#########################################################################
# Internal code
from dataclasses import dataclass
import numpy as np
from copy import deepcopy
from matplotlib.patches import Polygon, FancyArrowPatch, Wedge, Rectangle
import logging

logger = logging.getLogger('FlatPlanner')
logging.basicConfig(level=logging.INFO)

# Typing
from matplotlib.axes import Axes
from typing import Union


object_types = {
    'room':         dict(color='black', lw=4),
    'furniture':    dict(color='green', lw=3),
    'doors':        dict(color='brown', lw=4),
    'door_frame':   dict(color='white', lw=4),
    'window':       dict(color='black', lw=10),
    'outlet':       dict(color='red', lw=2),
}


def get_rLT(v1, v2):
    '''Return direction along and perpendicular to the vector betwee ntwo vertices.
    
    Vectors are normalized, and the convention is right-handed, such that rL point along index and rT along thumb.'''
    rL = np.array([v2[0]-v1[0], v2[1]-v1[1]])
    rL /= np.linalg.norm(rL)
    rT = np.array([-rL[1], rL[0]])

    return rL, rT

def rot_mat(ang: float, radians=False):
    '''Rotation matrix for right-handed rotation on 2D plane.

    Parameters
    ----------
    ang: 
        Angle of rotation in degrees
    '''
    if not radians: ang = np.radians(ang)
    return np.array([
                [ np.cos(ang),-np.sin(ang)],
                [ np.sin(ang), np.cos(ang)]
            ], dtype=float)

def get_angle(v):
    phi = np.degrees( np.arctan2(v[1], v[0]) )
    if phi<0: phi += 360
    return phi
    
@dataclass
class Dmarker:
    v1_id: Union[int, tuple['Object', int]]
    v2_id: Union[int, tuple['Object', int]]
    offset: float = 0
    label_pos: str = 'b'
    color: str = 'black'
    _v1: tuple[float, float] = tuple()
    _v2: tuple[float, float] = tuple()

    def __post_init__(self):
        if self.label_pos not in 'bt':
            logger.error(f'{self.__class__.__name__} `labels_pos` argument can only be `b` or `t`')

        for vv in [self.v1_id, self.v2_id]:
            if isinstance(vv, tuple):
                obj, v_id = vv
                if v_id >= len(obj.vertices):
                    logger.error(f'Dmarker index too large: vid={v_id}, Object(otype={obj.otype} nverts={len(obj.vertices)})')
                    raise IndexError(f'Dmarker index too large: vid={v_id}, Object(otype={obj.otype} nverts={len(obj.vertices)})')

    def draw(self, ax: 'Axes'):
        '''Plot distance marker between two vertices
        
        Parameters
        ----------
        ax: matplotlib.Axes
            Axes on which to plot.
        v1, v2: tuple[float, float]
            Coordinate of vertices from start to end.
        dmarker_offset: float
            Offset the marker position in the direction perpendicular to v1-v2 line.
        dlabel_offset: float (depracated?)
            Offset the label position further with respect to the marker.
        color: str='black' (optional)
            Color of the marker line.
        '''
        logger.debug(f'Making marker {self}')

        # longitudinal and transverse directions
        distance = np.linalg.norm(self._v2 - self._v1)
        if distance < 1e-8:
            logger.error(f'Marker length=0. {self}')
            raise ValueError(f'Marker length=0.')

        rL, rT = get_rLT(self._v1, self._v2)


        vs = self._v1 + (CONFIG['dmarker_offset']+self.offset)*rT
        label_offset = dict(t=-1, b=1)[self.label_pos] * CONFIG['dmarker_offset'] * 0.6
        v_label = vs + distance*rL/2 + rT*label_offset

        dm_lw = CONFIG['dmarker_width']
        marker = FancyArrowPatch(posA=vs, posB=vs+distance*rL, arrowstyle=f'|-|, widthA={dm_lw}, widthB={dm_lw}',
                                lw=dm_lw, color=self.color)
        ax.add_artist(marker)
        ax.text(v_label[0], v_label[1], f'{np.linalg.norm(distance):.0f}', 
                rotation=np.degrees(np.arctan2(rL[1], rL[0])),
                ha='center', va='center')

# VERTICES CONSTRUCTORS
class Verts(np.ndarray):
    '''Representation of vertices on 2D plane.
    
    Is a subclassed np.ndarray, of shape (N,2) with constructor
    shortcuts and  functionalities
    to transform the vertices on the plane.
    '''

    # Construction methods
    @classmethod
    def LINE(cls, length: float):
        vs = np.array([[0,0], [length,0]])
        return np.asarray(vs).view(cls)

    @classmethod
    def LIST(cls, ll: list):
        '''Construct vertices from a list of coordinates'''
        vs = np.array(ll, dtype=float)
        if vs.shape[1] != 2:
            raise ValueError(f'List must be shape (N,2) is: {vs.shape}')
        
        return np.asarray(vs).view(cls)

    @classmethod
    def RECT(cls, width: float, height: float):
        '''Construct vertices of a rectangle:
        
        Returns
        [ [0,0], [width,0], [width,height], [0,height]]
        '''
        vs = np.array([ [0,0], [width,0], [width,height], [0,height]], dtype=float)
        return np.asarray(vs).view(cls)
    
    @classmethod
    def CLIST(cls, commands, silent=False):
        command_types = ['l', 'r', 'u', 'd', 'ang', 'raw']
        vertices = [[0,0]]
        v_prev = [0,0]
        for command in commands:
            key = command[0]
            if key not in command_types:
                raise KeyError(f'Unrecognised command type {key}')
            
            if key == 'l':
                v_new = [v_prev[0]-command[1], v_prev[1]]
            if key == 'r':
                v_new = [v_prev[0]+command[1], v_prev[1]]
            if key == 'u':
                v_new = [v_prev[0], v_prev[1]+command[1]]
            if key == 'd':
                v_new = [v_prev[0], v_prev[1]-command[1]]
            if key == 'ang':
                ang = np.radians(command[1])
                dd = command[2]
                v_new = [v_prev[0]+dd*np.cos(ang), v_prev[1]+dd*np.sin(ang)]
            if key == 'raw':
                v_new = [command[1], command[2]]


            vertices.append(v_new)
            v_prev = v_new

        if not silent:
            if not np.allclose(v_new, [0,0], atol=1e-3, rtol=1e-5):
                logger.warning(f'Vertices not forming close polygon: {v_new!r}')

        vs = np.array(vertices, dtype=float)
        return np.asarray(vs).view(cls)
    
    @classmethod
    def OUTLET(cls, width: float, n: int=1):
        '''Electrical outlet represented by `n` triangles, each symbolizing
        a plug.
        
        Outer edges are under indices 0 and -1'''

        vs = np.linspace([0,0], [width,0], 2*n+1)
        vs[1::2,1] = CONFIG['outlet_length']

        return np.asarray(vs).view(cls)

    @classmethod
    def DOORS(cls, fwidth: float, dwidth: float, opening: float=90):
        '''Doors consist of frame, with outer width `fwidth` and doors
        of width `dwidth`. Each of them consists of two vertices, the four
        resulting vertices are collinear, and doors are centered within a frame.
        Doors `opening` angle can be positive or negative to indicate
        the direction of doors opening.

        Parameters
        ----------
        fwidth: float
            Width of the frame
        dwidth: float
            Width of the doors
        opening: float, (optional)
            Opening angle of the doors.

        Returns:
        vertices_doors: shape=(6,2)
            Frame with doors, first two vertices are outer edges of the frame,
            next two are edges of the doors, last two are sweeping through the
            opening of the doors.
            Door hinge is on the second of the doors vertices, v4=self.vertices[3]
        '''
        # Frame
        v1 = np.array([0,0])
        v2 = np.array([fwidth,0])
        
        vs = np.array([v1,v2], dtype=float)

        # Doors
        if dwidth>0:
            v3 = np.array([0,0])
            v4 = np.array([dwidth,0])

            v5 = rot_mat(opening) @ (np.array(v3)-np.array(v4)) + v4
            v6 = rot_mat(opening - np.sign(opening)) @ (np.array(v3)-np.array(v4)) + v4  # Helper vertex to understand over which path the ors are opening
        
            # Center doors within the frame
            cc = (fwidth-dwidth)/2
            v3 = v3 + [cc, 0]
            v4 = v4 + [cc, 0]
            v5 = v5 + [cc, 0]
            v6 = v6 + [cc, 0]

            vs = np.array([v1,v2,v3,v4,v5,v6], dtype=float)

        return np.asarray(vs).view(cls)

    # Utilities
    def __repr__(self):
        return 'Verts<' + super().__repr__() + '>'

    def translate(self, x: float=0, y: float=0, t: tuple[float,float]=()):
        '''Translate the vertices by [x,y] or t=(x,y).
        Providing both doesn't make sense, but in case, `t` has precedence.'''
        tr = [x,y]
        if len(t):
            tr = t

        return self + tr
    
    def rotate(self, angle, radians=False):
        '''Rotate vertices according to right-handed rotation around z.'''
        R = rot_mat(angle, radians=radians)
        return np.matmul(R, self.T).T
    
ID_COUNTER = 0
def assign_id():
    global ID_COUNTER
    ID_COUNTER += 1
    return ID_COUNTER
  

# MAIN OBJECTS
class Object():
    _id: int
    vertices: Verts
    otype: str
    plot_styles: dict
    dmarkers: list[Dmarker]
    
    def __init__(self, vertices: Verts, otype: str, color: str=None, dmarkers: list[Dmarker]=[]):
        '''Construct the object
        
        Parameters
        ----------
        vertices: Verts
            Corners of the `room` or `furniture`: N >= 3
            Edges of the `window` or `door_frame`. N=2
            Position of the `doors` and opening angle. N=2 or N=3. For N=2 default opening angle 180 deg.
        type: str
            One of the implemented object types.
        color: str (optional)
            Color of the object. If not provided takes default from the `object_type`.
        dmarkers: np.ndarray (M, 4) (optional)
            Distance markers, where each entry is (vid_1, vid_2, marker_transverse_offset, label_transverse_offset).
        '''
        assert otype in object_types.keys()

        self._id = assign_id()
        self.otype = otype

        self.plot_styles = deepcopy(object_types[otype])
        if color:
            self.plot_styles['color'] = color

        self.vertices = vertices

        # DMARKERS
        self.dmarkers = dmarkers
        for dd in self.dmarkers:
            dd.color = self.plot_styles['color']

        logger.info(f'Making object {self.otype}, id={self._id}')

    

    def __str__(self):
        return str(self.vertices)
    
    def __repr__(self):
        return f'<{self.__class__.__name__}, {self.otype}, ID={self._id}>'
    
    def __eq__(self, other):
        return self._id == other._id
    
    def copy(self) -> 'Object':
        '''Return deep copy of itself'''
        return deepcopy(self)
    

    def translate(self, x: float=0, y: float=0, t:tuple[float,float]=()):
        '''Translate the object'''
        self.vertices = self.vertices.translate(x=x, y=y, t=t)
        return self
    
    def rotate(self, angle: float, radians: bool=False):
        '''Rotate object vertices according to right-handed rotation around z.'''
        self.vertices = self.vertices.rotate(angle=angle, radians=radians)
        return self
    
    

    def get_area(self):
        '''Get area of the object in m2'''
        A = 0
        for v1,v2 in zip(self.vertices, np.roll(self.vertices, -1, axis=0)):
            A += (v1[0]*v2[1] - v1[1]*v2[0])/2

        return A / 1e4  # Convert to m2
    
    def get_angles(self):
        '''Get internal angles of the object.'''
        angles = []
        vs_angles = np.concatenate( ( [self.vertices[-2]], self.vertices[:-1], [self.vertices[0]]) ) 
        for n in range(len(vs_angles)-2):
            a1 = get_angle(vs_angles[n+2] - vs_angles[n+1])
            a2 = get_angle(vs_angles[n] - vs_angles[n+1])
            if a2<a1: a2+=360
            angles.append(a2-a1)

        return angles
    
    # MAIN DRAWER FUNCTION
    def draw(self, ax: 'Axes', hide_markers=False):
        logger.info(f'Drawing {self.otype}(id={self._id})')
        # Consider different shapes
        if self.otype == 'room':
            ax.add_artist( Polygon(self.vertices, lw=0, color='white') )
            ax.add_artist( Polygon(self.vertices, fill=None, hatch='++',
                                alpha=0.2, lw=0, color=self.plot_styles['color']))
            ax.add_artist( Polygon(self.vertices, fill=None,
                                **self.plot_styles))
        if self.otype == 'furniture':
            ax.add_artist( Polygon(self.vertices,
                                alpha=0.3, lw=0, color=self.plot_styles['color']))
            ax.add_artist( Polygon(self.vertices, fill=None,
                                **self.plot_styles))
        if self.otype == 'outlet':
            ax.add_artist( Polygon(self.vertices, lw=0, color=self.plot_styles['color']) )
        if self.otype == 'window':
            ax.add_artist( Polygon(self.vertices, **self.plot_styles))
        if self.otype == 'door_frame':
            ax.add_artist( Polygon(self.vertices[:2], **self.plot_styles))
        if self.otype == 'doors':
            ax.add_artist( Polygon(self.vertices[:2], lw=self.plot_styles['lw'], color='peru' ))
            ax.add_artist( Polygon(self.vertices[2:4], lw=self.plot_styles['lw'], color='white' ))
           
            r = np.linalg.norm(self.vertices[2]-self.vertices[3])
           
            # Sweep of the wedge seems to be always right-handed
            th1 = get_angle( self.vertices[2]-self.vertices[3] )
            th2 = get_angle( self.vertices[4]-self.vertices[3] )
            if (th2-th1)%360 > 180:
                th1, th2 = th2, th1

            logger.info(f'Drawing doors: th1={th1}, th2={th2}')
            ax.add_artist( Wedge(self.vertices[3], r, th1, th2, 
                                 alpha=0.7, color=self.plot_styles['color']) )
            # ax.add_artist( Wedge(self.vertices[3], r, th2, th3, 
            #                      alpha=0.7, color=self.plot_styles['color']) )

        # Add distance markers
        if not hide_markers:
            for dd in self.dmarkers:
                dd._v1 = self.vertices[dd.v1_id]
                dd._v2 = self.vertices[dd.v2_id]
                dd.draw(ax)

        # Adjust limits
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(min(xmin, self.vertices[:,0].min()-CONFIG['padding']),
                    max(xmax, self.vertices[:,0].max()+CONFIG['padding']))
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(min(ymin, self.vertices[:,1].min()-CONFIG['padding']),
                    max(ymax, self.vertices[:,1].max()+CONFIG['padding']))
        
        return
                     
class ObjectCollection():
    name: str
    objects: list[Object]
    dmarkers: list[Dmarker]

    def __init__(self, name: str, objects: list[Object], align: list[int]=[], dmarkers: list[Dmarker]=[]):
        logger.info(f'Seting up `ObjectCollection`: {name}')
        self.name = name
        self.objects = objects

        for a in align:
            self.align_vertices(*a)

        # Set up markers
        self.dmarkers = dmarkers

    def get_area(self):
        '''Get area of the room from the walls'''

        walls = [obj for obj in self.objects if obj.otype=='room']
        assert len(walls) == 1  # there should be only one room object
        return walls[0].get_area()

    def get_dims(self):
        '''Get main dimensions of the room from the extent of its walls'''

        walls = [obj for obj in self.objects if obj.otype=='room']
        assert len(walls) == 1  # there should be only one room object
        return np.ptp(walls[0].vertices, axis=0)    


    def translate(self, x: float=0, y: float=0, t: tuple[float,float]=()):
        for obj in self.objects:
            obj.translate(x=x, y=y, t=t)

        return self

    def rotate(self, angle: float, radians: bool=False):
        for obj in self.objects:
            obj.rotate(angle=angle, radians=radians)

        return self
           

    def draw(self, ax: 'Axes', hide_markers=False):
        # Background
        x = 1e5
        ax.add_artist( Rectangle((-x/2,-x/2),x,x, color='silver', zorder=-5001) )
        # ax.add_artist( Rectangle((-x/2,-x/2),x,x, fill=None, hatch='o',
        #                         alpha=0.2, color='black', zorder=-5001))
        
        # objects 
        for obj in self.objects:
            obj.draw(ax, hide_markers=hide_markers)

        # dmarkers
        if not hide_markers:
            for dd in self.dmarkers:
                print(dd)
                obj1, v_id1 = dd.v1_id
                obj2, v_id2 = dd.v2_id
                dd._v1 = obj1.vertices[v_id1]
                dd._v2 = obj2.vertices[v_id2]

                # set zero nominal offset
                dd.offset -= CONFIG['dmarker_offset']
            
                dd.draw(ax)
                # plot_dmarker(ax, v1=dd._v1, v2=dd._v2,
                #               offset=-CONFIG['dmarker_offset']+dd.offset, label_pos=dd.label_pos)

    def align_vertices(self, 
                       a1: tuple[Object, int, int],
                       a0: tuple[Object, int, int],
                       t: list[float]=None):
        '''Align object1 to object0 according to the align tuple `aN`.

        `objectN, oN_vid1, oN_vid2 = aN `

        The method will change the coordinates ob object1, such that:
        - vertices given by `oN_vid1` will get on top of each other
        - if `oN_vid2` are provided, the second provided vertex of
          object2 will be aligned with the line fiven by vertex 1 and 2
          of object0, i.e., object1 will be rotated.

        Apply final translation `t`. If single vertices are aligned, the translation is in XY coordinates,
        if alignment invon=lves rotation, translation is made in longitufdinal, transverse system wrt first vertex.
        
        Parameters
        ----------
        a1: tuple[Object, int, int]
            Alignment tuple of the object to be aligned
        a0: tuple[Object, int, int]
            Alignment tuple of the reference object
        t: [float, float]
            Final translation to be applied. 

        Returns
        -------
        tf, ang:
            Final translation and rotation angles.

        Notes
        -----
        - The return parameters allow to perform alignment by hand. 
          But one needs to remember the rotation is done around the alignment
          point:

          >>> tf, ang = ObjectCollection.align_vertices( (o1,0,1), (o0,0,1))
          >>> o3.translate(-o0.vertices[0]).rotate(ang).translate(tf)


        '''
        # logger.info(f'Aligning {a1!r} to {a0!r}')

        def unpack_alignment(a) -> tuple[Object, int, int]:
            ret = tuple()
            if len(a) == 1:
                ret = (a[0], None, None)
            if len(a) == 2:
                ret = (a[0], a[1], None)
            if len(a) == 3:
                ret = tuple(a)

            return ret
        
        obj1, obj1_v1, obj1_v2 = unpack_alignment(a1)
        obj0, obj0_v1, obj0_v2 = unpack_alignment(a0)

        if obj1_v1 is None:
            obj1_v1 = 0
            obj0_v1 = 0
        if obj0_v1 is None:
            obj0_v1 = 0

        r1 = obj1.vertices[obj1_v1]
        r0 = obj0.vertices[obj0_v1]

        # Rotation
        ang = 0
        if obj1_v2 is not None:
            r1_21 = obj1.vertices[obj1_v2] - obj1.vertices[obj1_v1]
            r0_21 = obj0.vertices[obj0_v2] - obj0.vertices[obj0_v1]

            # `ang` says how much obj1 is rotated wrt obj0
            ang = np.arctan2(r1_21[1], r1_21[0]) - np.arctan2(r0_21[1], r0_21[0])

        rot = rot_mat(-ang, radians=True)

        # Translation
        tf = [0, 0]
        if t is not None:
            tf = t

            if len(a0) == 3:
                rL, rT = get_rLT(obj0.vertices[obj0_v1], obj0.vertices[obj0_v2])
                tf = rL*t[0] + rT*t[1]

        new_vertices = [rot@(v-r1)+r0+tf for v in obj1.vertices]
        obj1.vertices = np.array(new_vertices).view(Verts)

        return r0+tf, np.degrees(-ang)
    
    def align_by_doors(self: 'ObjectCollection', 
                       d0: Object, d1: Object, 
                       shift: float=0, 
                       reverse_door1: bool=False):
        '''Align the room given the position of connecting doors.
        
        Returns
        -------
        Surface of the threshold.
        '''
        d0_, d1_ = d0.copy(), d1.copy()
        
        d0_fw = np.linalg.norm(d0_.vertices[1]-d0_.vertices[0])
        d1_fw = np.linalg.norm(d1_.vertices[1]-d1_.vertices[0])

        door_centering = -(d1_fw-d0_fw) / 2
        if reverse_door1:
            d1_.vertices = d1_.vertices[:2][::-1]
            door_centering += np.linalg.norm(d1_.vertices[1]-d1_.vertices[0])

        logger.info(f'Door d0={d0_.vertices[0]} d1={d1_.vertices[0]}')
        door_aligner = ObjectCollection(name='door_aligner', objects=[d0_, d1_])

        translation, rot_angle = door_aligner.align_vertices(a1=(d1_, 0,1), 
                                                            a0=(d0_, 0,1), t=[door_centering,shift])
        logger.info(f'Door tr={translation} ang={rot_angle}')

        self.translate(t=-d1.vertices[0]).rotate(rot_angle).translate(t=translation) #.rotate(rot_angle).translate(t=translation)

        return min(d0_fw, d1_fw)*np.abs(shift)*1e-4

#########################################################
# Dirty tests


def make_LivingRoom():
    lr_ang1 = 191.359
    walls_dims = [
            ['r', 455],
            ['u', 330],
            ['ang', 270-lr_ang1, 124],
            ['ang', 360-lr_ang1, 489],
            ['d', 548-0.117],
    ]
    dmarkers = [
        Dmarker(0, 1, 0),
        Dmarker(1, 2, -20),
        Dmarker(2, 3, 0),
        Dmarker(4, 3, 40, 't'),
        Dmarker(4, 5, 0),
    ]
    walls = Object(otype='room', vertices=Verts.CLIST(walls_dims), dmarkers=dmarkers)
    
    print(f'Living Room area={walls.get_area()} m2')
    print(f'Living Room angles={walls.get_angles()}')

    window = Object(otype='window', vertices=Verts.LINE(187))

    dH = Object(otype='doors', vertices=Verts.DOORS(89, 120, 90), color='brown')
    dK = Object(otype='door_frame', vertices=Verts.LIST([[0,0], [96, 0]]))
    dB = Object(otype='doors', vertices=Verts.DOORS(90, 86, -90), color='brown')

    # g1 = Object(object_type='outlet', vertices=v_rect(15,15)+[140,0], color='red')
    # g2 = Object(object_type='outlet', vertices=v_rect(8,15)+[485-78,0], color='red')

    couch_dims = [
            ['r', 252],
            ['d', 162],
            ['l', 110],
            ['u', 60],
            ['l', 252-110],
    ]
    couch = Object(otype='furniture', vertices=Verts.CLIST(couch_dims), color='orange',
                 dmarkers=[Dmarker(0,1,0), Dmarker(1,2,0)])


    return ObjectCollection(name='Living Room', objects=[walls, dH, dK, dB, couch], 
                            align=[
                                # [(window, 0, 1), (walls, 1, 2), [61,0]],
                                [(dH, 1, 0), (walls, 4, 5), [134, 0]],
                                [(dK, 1, 0), (walls, 3, 4), [49, 0]],
                                [(dB, 0, 1), (walls, 2, 3), [17, 0]],
                                ],
                            dmarkers=[
                            ])
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    lr = make_LivingRoom()
    lr.draw(ax)

    fig.savefig('debug.png')
