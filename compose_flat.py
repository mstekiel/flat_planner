CONFIG = dict(
universal_scale = True,
)

from cProfile import label
from matplotlib import tight_layout
import numpy as np
import matplotlib.pyplot as plt

from flat_planner import Dmarker, Verts, Object, ObjectCollection

import logging

logger = logging.getLogger('FlatComposer')
logging.basicConfig(level=logging.DEBUG)

def close_room():
    def residuals(ha1=100, ha2=90):
        # ha1 = 101.3
        # ha2 = 92.2
        walls_dims = [
            ['r', 123+77+77],
            ['u', 99+87+142],
            ['ang', 270-ha1, 7+102+110],
            ['ang', 270-ha1+180-ha2, 106+90+74],
            ['d', 11+89+9]
        ]

        vs = Verts.CLIST(walls_dims)
        return np.linalg.norm(vs[-1] - vs[0])
    
    # Simulation parameters
    N = 40
    ha1 = 101.50755
    ha2 = 91.85544
    p1, p2 = np.meshgrid(ha1 + 0.0001*np.linspace(-1,1,N), 
                         ha2 + 0.0001*np.linspace(-1,1,N))
    

    R = np.full(p1.shape, np.inf)
    for it in np.ndindex(R.shape):
        R[it] = residuals(p1[it], p2[it])

    optD = 2
    if optD == 2:
        fig, ax = plt.subplots()

        img = ax.pcolormesh(p1,p2, R, cmap='jet')
        fig.colorbar(img)
        plt.show()

def make_Hall(return_doors=False):
    ha1 = 101.50755
    ha2 = 91.85544
    walls_dims = [
        ['r', 123+77+77],
        ['u', 99+87+142],
        ['ang', 270-ha1, 7+102+110],
        ['ang', 270-ha1+180-ha2, 106+90+74],
        ['d', 11+89+9]
    ]
    walls = Object(otype='room', vertices=Verts.CLIST(walls_dims), 
                   dmarkers=[Dmarker(4,0)])
    print(f'Hall angles: {walls.get_angles()}')


    dM = Object(otype='doors', vertices=Verts.DOORS(102, 98, -90))
    dO = Object(otype='door_frame', vertices=Verts.LIST([[0, 0], [0,90]]))
    dBe= Object(otype='door_frame', vertices=Verts.LIST([[0, 0], [0,89]]))
    dBa= Object(otype='door_frame', vertices=Verts.LIST([[0, 0], [0,77]]))
    dL = Object(otype='door_frame', vertices=Verts.LIST([[0, 0], [0,87]]))

    dmarkers = [
        Dmarker(0, 2, 20),
    ]
    tree = Object(otype='furniture', vertices=Verts.RECT(80,80).translate(x=100), dmarkers=dmarkers)

    ret = ObjectCollection(name='Hall', objects=[walls, dM,dO,dBe,dBa,dL, tree], 
                            align=[
                                [(dM, 1,0), (walls,3,2), [7, 0]],
                                [(dO, 0,1), (walls,4,3), [106, 0]],
                                [(dBe,0,1), (walls,5,4), [11, 0]],
                                [(dBa,0,1), (walls,0,1), [123, 0]],
                                [(dL, 0,1), (walls,1,2), [99, 0]],
                                [(tree, 1), (walls,1)],
                                ],
                            dmarkers=[
                                Dmarker((walls,0), (dBa, 0), -20),
                                Dmarker((dBa, 1), (walls,1), -20),
                                Dmarker((walls,1), (dL, 0), -20),
                                Dmarker((dL, 1), (walls,2), -20),
                                Dmarker((dM, 0), (walls,2), 20, 't'),
                                Dmarker((walls,3), (dO, 1), -20),
                                Dmarker((dO, 0), (walls,4), -20),
                            ])
    
    if return_doors:
        ret = (ret, [dBe, dBa, dL, dO])

    return ret

def make_Kitchen(return_doors=False):
    # Kitchen
    walls_dims = [
            ['r', 111],
            ['u', 91],
            ['r',12],
            ['d', 91,],
            ['r', 613],
            ['u', 184+62],
            ['l', 613],
            ['d', 78,],
            ['l', 12],
            ['u', 78],
            ['l', 67],
            ['d', 62],
            ['l', 44],
            ['d', 184]
    ]
    dmarkers = [
        Dmarker(0,1),
        Dmarker(11,10, offset=40, label_pos='t'),
        Dmarker(11,12),
        Dmarker(13,12, offset=100, label_pos='t'),
        Dmarker(13,14),
    ]
    walls = Object(otype='room', vertices=Verts.CLIST(walls_dims), dmarkers=dmarkers)

    dmarkers = [
        Dmarker(3,2),
    ]
    # fridge = Object(object_type='furniture', vertices=v_rect(100,100)+100, dmarkers=dmarkers)
    blatt1 = Object(otype='furniture', vertices=Verts.RECT(287,60), color='pink', dmarkers=dmarkers)
    blatt2 = Object(otype='furniture', vertices=Verts.RECT(304,60), color='pink', dmarkers=dmarkers)
    doors = Object(otype='doors', vertices=Verts.DOORS(90, 86, -170).translate(435), color='brown')
    window = Object(otype='window', vertices=Verts.LINE(187))

    o1 = Object(otype='outlet', vertices=Verts.OUTLET(8,1).translate(601))
    o2 = Object(otype='outlet', vertices=Verts.OUTLET(8,1))
    
    align_hardware = [
        [(blatt1, 0), (walls, 4)],
        [(blatt2, 3), (walls, 7)],
        [(window, 0,1), (walls, 5,6), [37.5,0]],
        [(o2, 0,-1), (walls, 6,7), [120,0]],
    ]
    dmarkers_hardware = [
        Dmarker((doors,1), (walls, 5), offset=-40), # 211
        Dmarker((blatt1, 1), (doors,0), offset=-20), # 25
        Dmarker((blatt2, 2), (walls,6), offset=40, label_pos='t'), # 310
        Dmarker((walls, 5), (window,0), offset=-20), # 37.5
        Dmarker((window,1), (walls, 6), offset=-20), # 21
        Dmarker((doors, 1), (o1,0), offset=-20), # 84
        Dmarker((o1,-1), (walls, 5), offset=-20), # 84
        Dmarker((blatt2, 2), (o2,-1), offset=10, label_pos='t'), #
        Dmarker((o2,0), (walls, 6), offset=10, label_pos='t'), #
        ]

    table = Object(otype='furniture', vertices=Verts.RECT(100,150).translate(590,50), color='brown', dmarkers=[Dmarker(3,2), Dmarker(0,3)])
    c1 = Object(otype='furniture', vertices=Verts.RECT(40,40), color='brown', dmarkers=[Dmarker(3,2)])
    c2 = Object(otype='furniture', vertices=Verts.RECT(40,40), color='brown')
    c3 = Object(otype='furniture', vertices=Verts.RECT(40,40), color='brown')
    c4 = Object(otype='furniture', vertices=Verts.RECT(40,40), color='brown')

    align_software = [
        [(c1, 2), (table, 3)],
        [(c2, 3), (table, 2)],
        [(c3, 0), (table, 1)],
        [(c4, 1), (table, 0)],
    ]
    
    ret = ObjectCollection(name='Kitchen', objects=[walls, window, doors, o1,o2, blatt1,blatt2, table, c1,c2,c3,c4], 
                           align=align_hardware+align_software,
                           dmarkers=dmarkers_hardware)
    
    if return_doors:
        ret = (ret, [doors])

    return ret

def make_Bedroom(return_doors=False):
    bd_ang1 = 100.02
    bd_ang2 = 93.28
    # ba = f'BEDROOM ANGLES'
    # ba+= f'BEDROOM ANGLES'
    # print('BEDROOM ANGLES')
    walls_dims = [
            ['r', 485-0.2232],
            ['u', 345+0.2222],
            ['ang', 270-bd_ang1, 393],
            ['ang', 450-bd_ang1-bd_ang2, 425],
    ]
    dmarkers = [
        Dmarker(0, 1, -10),
        Dmarker(1, 2, -10),
        Dmarker(3, 2, 40, 't'),
        Dmarker(3, 4,-10),
    ]
    walls = Object(otype='room', vertices=Verts.CLIST(walls_dims), dmarkers=dmarkers)
    
    print('Bedroom angles: ', walls.get_angles())

    window = Object(otype='window', vertices=Verts.LINE(187))
    doors = Object(otype='doors', vertices=Verts.DOORS(90, 86, -bd_ang1), color='brown')

    g1 = Object(otype='outlet', vertices=Verts.OUTLET(15,2).translate(140), color='red')
    g2 = Object(otype='outlet', vertices=Verts.OUTLET(8,1).translate(485-78), color='red')

    carpet = Object(otype='furniture', vertices=Verts.RECT(230, 170).translate(130, 100), color='cyan')

    bed = Object(otype='furniture', vertices=Verts.RECT(170,225).translate(160), color='dodgerblue',
                 dmarkers=[Dmarker(0,3), Dmarker(3,2)])

    pax = Object(otype='furniture', vertices=Verts.RECT(250,60), color='orange')
    pd1 = Object(otype='doors', vertices=Verts.DOORS(50, 49, 90), color='orange')

    ret = ObjectCollection(name='Bedroom', objects=[walls, window, doors, carpet, bed, g1,g2, pax, pd1], 
                            align=[
                                [(doors, 0, 1), (walls, 1, 2), [220, 0]],
                                [(window, 0,1), (walls, 3,4), [68,0]],
                                [(pax, 3,2), (walls, 3,2)],
                                [(pd1, 1,0), (pax, 1,0)],
                                ],
                            dmarkers=[
                                Dmarker((window, 1), (walls, 0), -10),
                                Dmarker((walls, 3), (window, 0),-10),
                                Dmarker((walls, 1), (doors, 0),-10),
                                Dmarker((doors, 1), (walls, 2),-10),
                                Dmarker((pd1, -1), (bed, 2), 0),
                                Dmarker((pax, 1), (doors, -1), 0),
                                Dmarker((walls, 0), (g1,0), -10,),
                                Dmarker((g2,-1), (walls, 1), -10),
                            ])
    
    if return_doors:
        ret = (ret, [doors])

    return ret

def make_LivingRoom(return_doors=False):
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
        Dmarker(4, 3, 60, 't'),
        Dmarker(4, 5, -20),
    ]
    walls = Object(otype='room', vertices=Verts.CLIST(walls_dims), dmarkers=dmarkers)
    
    print(f'Living Room area={walls.get_area()} m2')
    print(f'Living Room angles={walls.get_angles()}')

    window = Object(otype='window', vertices=Verts.LINE(187))

    dH = Object(otype='door_frame', vertices=Verts.DOORS(89, 120, 90))
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


    ### Now there are two oiptions for the setup of the room
    # SETUP1
    if False:
        align_addons = [[(couch, 1), (walls, 2)]]
        dmarkers_addons = Dmarker((dH, 0), (couch, 0))

        tw, th = 100, 150
        cw, cs = 40, 0
        table = Object(otype='furniture', vertices=Verts.RECT(tw, th)+[50, 50], color='peru',
                    dmarkers=[Dmarker(3, 2, 0)])
        
        c1 = Object(otype='furniture', vertices=Verts.RECT(cw,cw), color='peru')
        c2 = Object(otype='furniture', vertices=Verts.RECT(cw,cw), color='peru')
        c3 = Object(otype='furniture', vertices=Verts.RECT(cw,cw), color='peru')
        c4 = Object(otype='furniture', vertices=Verts.RECT(cw,cw), color='peru')

        align_addons.extend([
            [(c1, 0,1), (table, 0,1), [-cw-cs, 0]],
            [(c2, 0,1), (table, 0,1), [tw+cs, 0]],
            [(c3, 3,2), (table, 3,2), [-cw-cs, 0]],
            [(c4, 3,2), (table, 3,2), [tw+cs, 0]],
            # [(c2, 2,3), (table, 3,2)],
            # [(c3, 0,1), (table, 1,0)],
            # [(c4, 3,2), (table, 2,3)],
            ])
        addons = [table, c1,c2,c3,c4]
    else:
        # SETUP 2

        tw, th = 80, 200
        cw, cs = 40, 0
        table = Object(otype='furniture', vertices=Verts.RECT(tw, th)+[320, 100], color='peru',
                    dmarkers=[Dmarker(3, 2), Dmarker(0,3)])
        
        c1 = Object(otype='furniture', vertices=Verts.RECT(cw,cw), color='peru')
        c2 = Object(otype='furniture', vertices=Verts.RECT(cw,cw), color='peru')
        c3 = Object(otype='furniture', vertices=Verts.RECT(cw,cw), color='peru')
        c4 = Object(otype='furniture', vertices=Verts.RECT(cw,cw), color='peru')

        TV = Object(otype='furniture', vertices=Verts.RECT(150, 50).translate(x=50), color='gray')

        dmarkers_addons = [
            Dmarker((walls, 0), (couch, 5), offset=-10),
            Dmarker((couch, 1), (table, 3)),
        ]
        align_addons = [
            [(couch, 5, 0), (walls, 5, 4), [200,0]],
            [(c1, 0,1), (table, 0,1), [-cw-cs, 0]],
            [(c2, 0,1), (table, 0,1), [tw+cs, 0]],
            [(c3, 3,2), (table, 3,2), [-cw-cs, 0]],
            [(c4, 3,2), (table, 3,2), [tw+cs, 0]],
            # [(c2, 2,3), (table, 3,2)],
            # [(c3, 0,1), (table, 1,0)],
            # [(c4, 3,2), (table, 2,3)],
            ]
        
        addons = [table, c1,c2,c3,c4, TV]

    ret =  ObjectCollection(name='Living Room', objects=[walls, dH, dK, dB, couch]+addons, 
                            align=[
                                # [(window, 0, 1), (walls, 1, 2), [61,0]],
                                [(dH, 1, 0), (walls, 4, 5), [134, 0]],
                                [(dK, 1, 0), (walls, 3, 4), [49, 0]],
                                [(dB, 0, 1), (walls, 2, 3), [17, 0]],
                                ]+align_addons,
                            dmarkers=dmarkers_addons+[
                                Dmarker((walls,4), (dK,0), 10, label_pos='t'),
                                Dmarker((walls,4), (dH,1), -10),
                            ]
                            )
    
    if return_doors:
        ret = (ret, [dH, dB, dK])

    return ret

def make_Bathroom(return_doors=False):
    clist = [
        ['r', 118],
        ['u', 30],
        ['r', 154],
        ['u', 170],
        ['l', 272],
        ['d', 200],
    ]

    walls = Object(otype='room', vertices=Verts.CLIST(clist),
                   dmarkers=[
                       Dmarker(1,2),
                       Dmarker(2,3, -30),
                       Dmarker(3,4, -30),
                       Dmarker(5,4, 60, 't'),
                       Dmarker(5,6, -20)
                   ])
    
    doors = Object(otype='doors', vertices=Verts.DOORS(fwidth=79, dwidth=75, opening=92))

    toilet = Object(otype='furniture', vertices=Verts.RECT(40,40), color='pink')
    sink = Object(otype='furniture', vertices=Verts.RECT(60,30), color='pink')
    bathtub = Object(otype='furniture', vertices=Verts.RECT(70,170), color='pink')
    heater = Object(otype='window', vertices=Verts.LINE(86), color='gray')



    ret = ObjectCollection(name='Bathroom', objects=[walls, doors, toilet,sink,bathtub,heater],
                           align=[
                               [(doors, 0), (walls, 5), [121,0]],
                               [(toilet, 0,1), (walls, 0,1), [40,0]],
                               [(sink, 0,1), (walls, 2,3), [8,0]],
                               [(bathtub, 2), (walls, 4)],
                               [(heater, 0,1), (walls, 5,6), [114,0]],
                           ],
                           dmarkers=[
                               Dmarker((walls,5), (doors,0), 10, 't'),
                               Dmarker((doors,1), (walls,4), 10, 't'),
                               Dmarker((walls,5), (heater,0), -10),
                           ])

    if return_doors:
        ret = (ret, [doors])

    return ret

def make_Office(return_doors=False):
    clist = [
        ['r', 418],
        ['u', 262],
        ['l', 418],
        ['d', 262]
    ]

    walls = Object(otype='room', vertices=Verts.CLIST(clist),
                   dmarkers=[
                    #    Dmarker(1,2),
                    #    Dmarker(2,3, 10),
                    #    Dmarker(3,4, -30),
                    #    Dmarker(4,5),
                    #    Dmarker(6,5, 40, label_pos='t')
                   ])
    
    doors = Object(otype='doors', vertices=Verts.DOORS(fwidth=90, dwidth=86, opening=90))

    o1 = Object(otype='outlet', vertices=Verts.OUTLET(width=15, n=2))
    o2 = Object(otype='outlet', vertices=Verts.OUTLET(width=15, n=2))

    window = Object(otype='window', vertices=Verts.LINE(99))

    clist = [
        ['r', 190],
        ['u', 77],
        ['l', 128],
        ['u', 67],
        ['l', 62],
        ['d', 144],
    ]
    couch = Object(otype='furniture', vertices=Verts.CLIST(clist), color='darkblue')
    pax = Object(otype='furniture', vertices=Verts.RECT(100, 58), color='pink')

    ret = ObjectCollection(name='Office', objects=[walls, doors, o1,o2, window, couch, pax],
                           align=[
                               [(doors, 0,1), (walls, 2,1), [106,0]],
                               [(o1, 0,-1), (walls, 0,1), [300-66,0]],
                               [(o2, 0,-1), (walls, 2,3), [300-95,0]],
                               [(window, 0,1), (walls, 3,4), [75,0]],
                               [(couch, 0), (walls, 0), [0,0]],
                               [(couch, 0), (walls, 0), [0,0]],
                               [(pax, 2), (walls, 2), [0,0]],
                           ],
                           dmarkers=[
                               Dmarker((walls,1), (doors,1), offset=-20),   # 66 cm
                               Dmarker((doors,0), (walls,2), offset=-20),   # 106 cm
                               Dmarker((o1,-1), (walls,1), offset=-20),      # 163 cm
                               Dmarker((walls,3), (o2,-1), offset=20, label_pos='t'),      # 192 cm
                               Dmarker((walls,3), (window,0), offset=-10),  # 75 cm
                               Dmarker((window,1), (walls,4), offset=-10),  # 88 cm
                           ])

    if return_doors:
        ret = (ret, [doors])

    return ret

def plot_rooms():
    # rooms = [make_Office()]
    rooms = [make_Office(), make_Bathroom(), make_Hall(), make_Kitchen(), make_Bedroom(), make_LivingRoom()]

    for room in rooms:
        print(f'Making plan: {room.name}')
        fig, ax = plt.subplots(figsize=(29.7/2.52, 21.0/2.52), tight_layout=True)
        ax.set_title(room.name, fontdict=dict(fontsize=16))
        ax.set_axisbelow(True)
        ax.grid(zorder=-4000)

        if CONFIG['universal_scale']:
            uw = 900
            uh = 550
            rc = room.get_dims()
            logger.info(f'Center of {room.name} = {rc}')
            ax.set_xlim( (rc[0]-uw)/2, (rc[0]+uw)/2)
            ax.set_ylim( (rc[1]-uh)/2, (rc[1]+uh)/2)

        ax.set_aspect(1)

        room.draw(ax)
        # tree.paint(ax)

        output_name = f'./b4-{room.name.replace(" ","")}.png'
        fig.savefig(output_name)


def plot_flat():
    Hall, Hall_doors = make_Hall(return_doors=True)
    Kitchen, Kitchen_doors = make_Kitchen(return_doors=True)
    Bedroom, Bedroom_doors = make_Bedroom(return_doors=True)
    LivingRoom, LivingRoom_doors = make_LivingRoom(return_doors=True)
    Bathroom, Bathroom_doors = make_Bathroom(return_doors=True)
    Office, Office_doors = make_Office(return_doors=True)
    
    A_t = 0

    # Plotting
    margins = dict()
    fig_main, ax_main = plt.subplots(figsize=(29.7/2.52, 21.0/2.52), tight_layout=True)
    ax_main.set_title('Bürgerplatz 4', fontdict=dict(fontsize=16, fontweight=15))
    ax_main.set_axisbelow(True)
    ax_main.grid(zorder=-4000)
    ax_main.set_aspect(1)

    # Align rooms
    Bedroom.draw(ax_main, hide_markers=True)


    A_tH = Hall.align_by_doors(d0=Bedroom_doors[0], d1=Hall_doors[0], shift=-11)
    Hall.draw(ax_main, hide_markers=True)

    A_tL = LivingRoom.align_by_doors(d0=Hall_doors[2], d1=LivingRoom_doors[0], shift=-29)
    LivingRoom.draw(ax_main, hide_markers=True)

    A_tK = Kitchen.align_by_doors(d0=LivingRoom_doors[2], d1=Kitchen_doors[0], shift=29)
    Kitchen.draw(ax_main, hide_markers=True)

    A_tB = Bathroom.align_by_doors(d0=Hall_doors[1], d1=Bathroom_doors[0], shift=-11)
    Bathroom.draw(ax_main, hide_markers=True)

    A_tO = Office.align_by_doors(d0=Hall_doors[3], d1=Office_doors[0], 
                          reverse_door1=True, shift=20)
    Office.draw(ax_main, hide_markers=True)

    # plt.show()
    fig_main.savefig('./b4.png')


    A_Be = Bedroom.get_area()
    A_Ba = Bathroom.get_area()
    A_H = Hall.get_area()
    A_LR = LivingRoom.get_area()
    A_K = Kitchen.get_area()
    A_O = Office.get_area()
    A_T = A_tH+A_tK+A_tL+A_tO+A_tB
    A_total = A_Be + A_Ba + A_H + A_LR + A_K + A_O + A_T
    print(f'Areas of rooms [m2]')
    print(f'  Thresholds = {A_T:.2f}')
    print(f'  Bedroom    = {A_Be:.2f}')
    print(f'  Bathroom   = {A_Ba:.2f}')
    print(f'  Hall       = {A_H:.2f}')
    print(f'  Living Room= {A_LR:.2f}')
    print(f'  Kitchen    = {A_K:.2f}')
    print(f'  Office     = {A_O:.2f}')
    print(f'  --------------------')
    print(f'  Total      = {A_total:.2f} ')


if __name__ == '__main__':
    plot_rooms()
    plot_flat()

    # Kitchen = make_Kitchen()
    # print(Kitchen.get_center())