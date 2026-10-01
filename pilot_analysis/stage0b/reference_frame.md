# Stage 0b — reference-frame test

BASE training instances only. Features are computed over each instance's own annotated extent from the GT boxes. Each feature is signed so that **AUC > 0.5 means it agrees with its hypothesis** and 0.5 means no information. CIs resample whole videos.

Heading features use the displacement of the box centre between the first and last frames of the extent, projected onto the mean subject→object direction.

## front / behind (6391 instances)

| stratum | feature | n | n front/left | AUC | 95% CI (video bootstrap) |
|---|---|---:|---:|---:|---|
| all | (i) log area ratio subj/obj  [camera: front=larger] | 6391 | 3369 | 0.734 | 0.687–0.778 |
| all | (ii) bottom-edge y diff  [camera: front=lower in image] | 6391 | 3369 | 0.894 | 0.864–0.922 |
| all | (iii) subject velocity . subj->obj  [heading: front=moving away] | 6391 | 3369 | 0.523 | 0.494–0.552 |
| all | (iii-b) object velocity . obj->subj  [heading: front=obj approaches] | 6391 | 3369 | 0.520 | 0.492–0.550 |
| subject verb stationary (stand/sit/lie/stop) | (i) log area ratio subj/obj  [camera: front=larger] | 3195 | 1655 | 0.765 | 0.716–0.811 |
| subject verb stationary (stand/sit/lie/stop) | (ii) bottom-edge y diff  [camera: front=lower in image] | 3195 | 1655 | 0.936 | 0.904–0.961 |
| subject verb stationary (stand/sit/lie/stop) | (iii) subject velocity . subj->obj  [heading: front=moving away] | 3195 | 1655 | 0.492 | 0.446–0.531 |
| subject verb stationary (stand/sit/lie/stop) | (iii-b) object velocity . obj->subj  [heading: front=obj approaches] | 3195 | 1655 | 0.495 | 0.457–0.532 |
| subject verb moving | (i) log area ratio subj/obj  [camera: front=larger] | 2639 | 1423 | 0.699 | 0.642–0.751 |
| subject verb moving | (ii) bottom-edge y diff  [camera: front=lower in image] | 2639 | 1423 | 0.842 | 0.789–0.886 |
| subject verb moving | (iii) subject velocity . subj->obj  [heading: front=moving away] | 2639 | 1423 | 0.560 | 0.520–0.604 |
| subject verb moving | (iii-b) object velocity . obj->subj  [heading: front=obj approaches] | 2639 | 1423 | 0.547 | 0.502–0.594 |
| subject measurably moving (>2% of diagonal) | (i) log area ratio subj/obj  [camera: front=larger] | 4319 | 2268 | 0.723 | 0.670–0.769 |
| subject measurably moving (>2% of diagonal) | (ii) bottom-edge y diff  [camera: front=lower in image] | 4319 | 2268 | 0.880 | 0.837–0.913 |
| subject measurably moving (>2% of diagonal) | (iii) subject velocity . subj->obj  [heading: front=moving away] | 4319 | 2268 | 0.528 | 0.490–0.567 |
| subject measurably moving (>2% of diagonal) | (iii-b) object velocity . obj->subj  [heading: front=obj approaches] | 4319 | 2268 | 0.514 | 0.475–0.549 |
| object measurably moving (>2% of diagonal) | (i) log area ratio subj/obj  [camera: front=larger] | 4313 | 2250 | 0.724 | 0.669–0.768 |
| object measurably moving (>2% of diagonal) | (ii) bottom-edge y diff  [camera: front=lower in image] | 4313 | 2250 | 0.880 | 0.840–0.915 |
| object measurably moving (>2% of diagonal) | (iii) subject velocity . subj->obj  [heading: front=moving away] | 4313 | 2250 | 0.519 | 0.482–0.555 |
| object measurably moving (>2% of diagonal) | (iii-b) object velocity . obj->subj  [heading: front=obj approaches] | 4313 | 2250 | 0.524 | 0.486–0.558 |

## left / right (8254 instances)

| stratum | feature | n | n front/left | AUC | 95% CI (video bootstrap) |
|---|---|---:|---:|---:|---|
| all | signed horizontal offset  [camera: left=subject further left] | 8254 | 4124 | 0.971 | 0.956–0.982 |
| subject verb stationary (stand/sit/lie/stop) | signed horizontal offset  [camera: left=subject further left] | 4373 | 2207 | 0.971 | 0.950–0.986 |
| subject verb moving | signed horizontal offset  [camera: left=subject further left] | 3158 | 1535 | 0.970 | 0.952–0.983 |
