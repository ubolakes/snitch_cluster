<%
    nr_clusters = cfg['nr_s1_quadrant']*cfg['s1_quadrant']['nr_clusters']
    base_hartid = cfg['cluster']['cluster_base_hartid']
    nr_iter = 2
%>
[
% for i in range(0, nr_clusters):
% for j in range(0, 8):
    {
        "thread": "${f'hart_{base_hartid + 9*i + j}'}",
        "roi": [
            {"idx": 1, "label": "alloc"},
% for iter in range(nr_iter):
<%
    if j == 0:
        roi_per_iter = 7
    else:
        roi_per_iter = 5
%>
            {"idx": ${3 + iter * roi_per_iter}, "label": "setup"},
            {"idx": ${4 + iter * roi_per_iter}, "label": "assignment"},
            {"idx": ${5 + iter * roi_per_iter}, "label": "barrier"},
            {"idx": ${6 + iter * roi_per_iter}, "label": "update"},
            {"idx": ${7 + iter * roi_per_iter}, "label": "barrier"},
% if j == 0:
            {"idx": ${8 + iter * roi_per_iter}, "label": "reduction"},
            {"idx": ${9 + iter * roi_per_iter}, "label": "barrier"},
% endif
% endfor
        ]
    },
% endfor
% if i == 0:
{
    "thread": "${f'dma_{base_hartid + 9*i + 8}'}",
    "roi": [
        {"idx": -3, "label": "samples"},
        {"idx": -2, "label": "centroids in"},
        {"idx": -1, "label": "centroids out"},
    ]
},
% else:
{
    "thread": "${f'dma_{base_hartid + 9*i + 8}'}",
    "roi": [
        {"idx": -2, "label": "samples"},
        {"idx": -1, "label": "centroids in"},
    ]
},
% endif
% endfor
]