---
layout: project_page
permalink: /

title: Text-guided 3D Human Garment Editing with Precise Localization and Cross-View Consistency
authors:
affiliations:
paper: 
video: 
code: https://github.com/ssr98-rgb/THGE-main
data:
---

<!-- Using HTML to center the abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Abstract</h2>
        <div class="content has-text-justified">
Recent advances in 2D diffusion models have substantially facilitated progress in 3D editing. Nevertheless, text-driven human garment editing remains largely underexplored, and existing methods often lead to inaccurate localization and multi-view inconsistencies. To address these issues, this work proposes Text-guided 3D Human Garment Editing (THGE), a dual-domain 2D–3D framework for precise localization and consistent editing. Specifically, 2D–3D co-localization with virtual Gaussian seeding is introduced to obtain robust target localization by merging garment masks before and after editing, while virtual Gaussians seeded along SMPL-X normals resolve the absence of corresponding Gaussians caused by garment extension or deformation. Second, SMPL-X-vertex guided cross-view consistent editing further enforces multi-view consistency without extra training by aggregating tokens of the same SMPL-X vertex across key views and propagating weighted features to other views. Finally, dual-domain 2D–3D Gaussian pruning is proposed to remove out-of-bound Gaussians by comparing masks rendered from labeled Gaussians with reference 2D masks, while an SDF-based human distance field constrains duplication and splitting to prevent Gaussians overflow. Experiments on multi-person and multi-garment scenarios demonstrate that THGE achieves superior visual quality and consistency over state-of-the-art methods.        </div>
    </div>
</div>

---

<h2 style="text-align:center; font-size:2rem; margin:24px 0 6px;">
  Overview
</h2>
<div style="text-align:center;">
<img src="{{ site.baseurl }}/static/image/framework.png" 
     alt="Framework" 
     style="max-width:100%; border-radius:12px; box-shadow:0 0 10px rgba(0,0,0,0.1);">
</div>

<h2 style="text-align:center; font-size:2rem; margin:24px 0 6px;">
  Comparison to other 3D editing methods
</h2>

<p style="margin:0 0 8px; font-size:1rem; opacity:.85;">
  We provide qualitative comparison with other 3D editing methods.
</p>

<!-- 第一张图 -->
<div style="text-align:center; margin:0 0 8px;">
  <img src="{{ site.baseurl }}/static/image/man1.png"
       alt="Man example"
       style="display:block; margin:0 auto; max-width:100%; height:auto; border-radius:12px; box-shadow:0 0 8px rgba(0,0,0,.12);">
</div>

<!-- 第二张图 -->
<div style="text-align:center; margin:0;">
  <img src="{{ site.baseurl }}/static/image/woman.png"
       alt="Woman example"
       style="display:block; margin:0 auto; max-width:100%; height:auto; border-radius:12px; box-shadow:0 0 8px rgba(0,0,0,.12);">
</div>

## Citation
```

```
