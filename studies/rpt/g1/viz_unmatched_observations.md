# RPT G1, descriptive: what the 20 sampled "no GT object nearby" SAM 3 tracks show

2026-10-29 · after the verdict (FAIL, A29d) · decides nothing · HELD in-sample, not a test result

**What this is.** A visual reading of the 20 images in `studies/rpt/g1/viz_unmatched/`.
- **The sample:** 20 of SAM 3's 196 unmatched confident tracks with best cut-vIoU < 0.1 to every GT
  run (`g1_report.md` §6.3 and §6.5), drawn with `numpy.random.default_rng(0)`.
- **The images:** each track is drawn in red on its middle frame, with its class and score. The GT
  runs covering that frame are in green.
- **How I looked:**
  - each drawn image at full size;
  - then the **raw frame** around each box, enlarged, with the outline drawn outside the box.

  In the drawn images, the red outline and label cover most of a small box and hide what is inside.
  The readings below come from the raw frames.
- **Checked by the thesis author,** who corrected images 03, 08 and 18. The raw-frame check then
  changed 04, 06, 07, 12, 17, 19 and 20.

**Limits:**
- This is a judgement from one frame per track, not a measurement.
- 20 tracks are a small sample of 196. A group of 6 out of 20 (30%) is compatible with roughly
  12–54% of the 196.

## Summary

| what the red box shows | images | count |
|---|---|---|
| a real object of the named class that has no GT box on that frame | 02, 04, 07, 09, 10, 12 | 6 |
| probably a real object of the named class | 13 | 1 |
| a real object, but not of the named class | 01 | 1 |
| a small piece of a larger object | 03, 05, 08 | 3 |
| no visible object | 11, 15, 16, 17, 18 | 5 |
| unclear: too small or too ambiguous to tell | 06, 14, 19, 20 | 4 |
| **all** | | **20** |

**Readings:**
- **Unannotated real objects explain about a third of the sample:** 6 clear cases, plus 1 probable.
- **About half are SAM 3 errors:**
  - boxes on nothing (5);
  - small pieces of larger objects (3), e.g. a piece of a person's back as "elephant" at score 0.93;
  - a wrong class on a real object (1), a plastic lawn flamingo as "bird".
- **4 are unclear.**
- **A recurring artefact:** tiny boxes touching the frame edge that last a long time. Images 11, 14,
  16 and 17 are 4–19 px wide, touch the left or right edge, and persist for 83–149 frames. Three of
  them are on plain background or haze.

## Per image

Size is the red box's width × height in pixels. Length is the track's length in frames.

| # | video | class, score | size; length | GT runs on frame | what the red box shows | group |
|---|---|---|---|---|---|---|
| 01 | `…00025029` | bird 0.87 | 78×69; 34 | 2 (person, dog) | a pink plastic lawn flamingo, cut by the right edge | real object, not the class |
| 02 | `…00027003` | whale 0.62 | 348×190; 10 | 0 | an orca leaping out of the water (blurry footage); no GT run covers frames 81–90 | real object of the class |
| 03 | `…00071019` | bicycle 0.77 | 56×57; 162 | 2 (person, bicycle) | the rider's backside, inside the GT boxes | piece of an object |
| 04 | `…00161016` | person 0.60 | 5×15; 114 | 2 (person, skateboard) | a distant person standing at the far side of the skate park | real object of the class |
| 05 | `…00234001` | sheep 0.80 | 46×31; 16 | 2 (both labelled antelope) | the top of a bighorn ram's horn at the corner of its GT box, mostly background | piece of an object |
| 06 | `…00265074` | ball 0.64 | 35×24; 116 | 0 | a bright yellow-green object inside a dark flying shape (a bag or cloth?) beside a mountain biker; possibly a ball | unclear |
| 07 | `…00339003` | car 0.71 | 32×19; 8 | 0 | the windshield and roof of a parked car in the background | real object of the class |
| 08 | `…00349000` | elephant 0.93 | 80×200; 30 | 0 | a small piece of the back of a person close to the camera; the elephant in the frame is elsewhere | piece of an object |
| 09 | `…00415008` | person 0.81 | 21×66; 11 | 2 (person, motorcycle) | a pedestrian in the background | real object of the class |
| 10 | `…00467000` | car 0.59 | 391×719; 52 | 0 | the rear of a car filling the left third of the frame | real object of the class |
| 11 | `…00480000` | sofa 0.66 | 8×33; 149 | 0 | the top-left corner of the frame, plain orange background | no visible object |
| 12 | `…00588000` | fox 0.79 | 23×28; 150 | 0 | a fox sitting in a field, far from the camera | real object of the class |
| 13 | `…00691000` | person 0.72 | 72×71; 5 | 0 | a pink-red shape inside a speedboat, seen from above; looks like a person in red clothing | probably the class |
| 14 | `…00721000` | person 0.81 | 4×24; 108 | 0 | a 4-pixel sliver of light background at the right edge, near a standing person | unclear |
| 15 | `…00989000` | person 0.62 | 132×113; 97 | 2 (horses) | a bush with red flowers beside two horses; no person | no visible object |
| 16 | `ILSVRC2015_val_00028003` | bird 0.62 | 19×24; 114 | 5 (whales, boat) | plain haze at the left edge; the seagulls in the sky are elsewhere | no visible object |
| 17 | `ILSVRC2015_val_00028003` | bird 0.70 | 11×10; 83 | 5 (whales, boat) | plain haze at the left edge | no visible object |
| 18 | `ILSVRC2015_val_00028003` | bird 0.74 | 28×14; 7 | 0 | a faint smudge in the haze; a seagull flies just outside the box, to its left | no visible object |
| 19 | `ILSVRC2015_val_00103000` | airplane 0.75 | 10×5; 58 | 0 | a small light shape just above a large parked airplane, apparently in the background | unclear |
| 20 | `ILSVRC2015_val_00103000` | airplane 0.73 | 6×6; 60 | 0 | a small light shape just above the same airplane | unclear |

**Notes:**
- **"GT runs on frame 0"** (12 of the 20) means no annotated object at all on that frame. HELD
  annotates only objects that take part in a relation, and only on some frames.
- **Image 05:** the GT calls both animals "antelope" (they look like bighorn sheep), and SAM 3's track
  is "sheep". The box covers only a corner of one GT box, so this is a misplaced piece, not a whole
  animal with a disputed name.
- **Image 05 has a small yellow rectangle near the top.** It is part of the video frame itself, not
  drawn by us.
- **Images 19 and 20:** the large airplane has no GT run on those frames either. Neither box covers
  it.
