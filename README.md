# Tool for depicting flat and room plans

## Functionalities

The main library is `flat_planner.py`. The `compose_flat.py` is a functional example to assemble the B4 apartment.

## TODO

- [ ] Subticks and labels on the main axes.
- [x] Dealing with doors connecting two rooms. Frame size can be different on either side of the threshold. Doors themselves are smaller than the frame. Do I implement frame and door size separately?
  - ~~How about implementing `doors` as an add-on to the `door_frame`? Or even implementing `door_frame` with an extra argument `add_doors` or step further `door_opening` float, that gives opening angle of the doors. Now I am set on th efact that doors only open to the inside of the room, and the opening angle sign, would determine the hinge point.~~
  - Implemented `doors` and `door_frame` separately. The distance between them is great for total dlat assembly.
- [x] Calculate total surface of the flat with the surface of thresholds -> Shoelace formula
- [x] Assembling a flat could be moved to the main module? *NO*