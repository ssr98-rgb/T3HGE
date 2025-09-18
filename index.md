---
layout: project_page
permalink: /

title: "T3HGE: High-fidelity Text-Driven 3D Human Garment Editing with Body Priors"
authors:
affiliations:
paper: 
video: 
code: 
data:
---

<!-- Code availability note -->
<div class="columns is-centered has-text-centered">
  <div class="column is-four-fifths">
    <div class="content has-text-justified" style="margin-bottom:20px;">
      <p><b>We will release the full code after the paper has been peer-reviewed and accepted.</b></p>
    </div>
  </div>
</div>

<!-- Using HTML to center the abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Abstract</h2>
        <div class="content has-text-justified">
While 3D Gaussian editing has seen substantial progress, text-driven human garment editing remains largely underexplored. Existing methods typically follow a paradigm that applies 2D editing techniques to multi-view rendered images and subsequently updates the 3D Gaussians based on these modified images. Such methods often suffer from distortion due to unintended region contamination and multi-view inconsistencies introduced by per-image independent editing. In this paper, we propose a high-fidelity Text-guided 3D Human Garment Editor (T3HGE), which is developed based on body priors derived from SMPL eXpressive (SMPL-X). T3HGE begins with seeding Gaussians along the normals of the SMPL-X model,  followed by 2D mask filtering to obtain precisely positioned Gaussians for editing. Multi-view consistency is enforced without additional training by aggregating tokens corresponding to the same SMPL-X vertex across key views. Furthermore, T3HGE integrates a Signed Distance Function (SDF)-based human distance field with 2D masks to constrain the duplication and splitting of Gaussians, thereby effectively preventing Gaussian overflow. Experiments on multi-person and multi-garment scenarios demonstrate that T3HGE outperforms existing state-of-the-art methods in both visual quality and cross-view consistency.        </div>
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

<p style="text-align:center; font-size:1rem; opacity:1; margin-top:8px; max-width:100%; margin-left:auto; margin-right:auto;">
  <b>Overall framework of THGE</b>, which consists of three components: 
  (1) 2D–3D Co-localization with Virtual Gaussian Seeding; 
  (2) SMPL-X-vertex Guided Cross-view Consistent Editing; and 
  (3) Dual-domain 2D–3D Gaussian Pruning.
</p>
<div style="margin-bottom:32px;"></div>



<h2 style="text-align:center; font-size:2rem; margin:40px 0 6px;">
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
<div style="margin-bottom:32px;"></div>



<h2 style="text-align:center; font-size:2rem; margin:40px 0 6px;">
  Quantitative Results
</h2>

<table style="margin:auto; border-collapse:collapse; text-align:center;">
  <thead>
    <tr>
      <th style="border:1px solid #ddd; padding:6px;">Method</th>
      <th style="border:1px solid #ddd; padding:6px;">CLIP Similarity</th>
      <th style="border:1px solid #ddd; padding:6px;">CLIP Directional Similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">GaussianEditor</td>
      <td style="border:1px solid #ddd; padding:6px;">0.2238</td>
      <td style="border:1px solid #ddd; padding:6px;">0.0230</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">DGE</td>
      <td style="border:1px solid #ddd; padding:6px;">0.2333</td>
      <td style="border:1px solid #ddd; padding:6px;">0.0624</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">EditSplat</td>
      <td style="border:1px solid #ddd; padding:6px;">0.2411</td>
      <td style="border:1px solid #ddd; padding:6px;">0.0910</td>
    </tr>
    <tr style="font-weight:bold;">
      <td style="border:1px solid #ddd; padding:6px;">THGE (Ours)</td>
      <td style="border:1px solid #ddd; padding:6px;">0.2543</td>
      <td style="border:1px solid #ddd; padding:6px;">0.1362</td>
    </tr>
  </tbody>
</table>

<p style="text-align:center; font-size:1rem; opacity:1; margin-top:8px;">
  <i>Quantitative Comparison.</i> CLIPdir: CLIP text-image direction similarity; 
  CLIPsim: CLIP text-image similarity. The best results are <b>highlighted in bold</b>.
</p>


<h2 style="text-align:center; font-size:2rem; margin:40px 0 6px;">
  Rendered Video Results
</h2>
<!-- === Four videos in one background, same height === -->
<style>
  .video-block {
    background:#f5f6f7;
    padding:18px;
    border-radius:14px;
    box-shadow:0 0 12px rgba(0,0,0,.08);
    margin:20px 0;
  }
  .video-row {
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:16px;  /* 间距比之前小 */
    align-items:start;
  }
  .video-row figure {
    text-align:center;
    margin:0;
  }
  /* 视频样式：统一高度 */
  .video-row video {
    display:block;       /* 避免基线问题 */
    width:100%;
    height:360px;   /* 固定高度，保持一致 */
    object-fit:cover;
    border-radius:10px;
  }
  /* 说明文字 */
  .video-row figcaption {
    margin-top:6px;
    font-size:14px;
    font-style:italic;
    color:#333;
  }
/* 盖掉主题对 figure 的默认外边距，避免第2~4个被往下推 */
.video-row figure { 
  margin: 0 !important;
}
.video-row figure + figure {
  margin-top: 0 !important;
}
</style>

<div class="video-block">
  <div class="video-row">
    <figure>
      <video autoplay loop muted playsinline>
        <source src="static/video/source.mp4" type="video/mp4">
      </video>
      <figcaption>Source</figcaption>
    </figure>
    <figure>
      <video autoplay loop muted playsinline>
        <source src="static/video/suit22.mp4" type="video/mp4">
      </video>
      <figcaption>"Make him in a suit"</figcaption>
    </figure>
    <figure>
      <video autoplay loop muted playsinline>
        <source src="static/video/shorts22.mp4" type="video/mp4">
      </video>
      <figcaption>"Make him in a shorts"</figcaption>
    </figure>
    <figure>
      <video autoplay loop muted playsinline>
        <source src="static/video/pants22.mp4" type="video/mp4">
      </video>
      <figcaption>"Turn his lower body into a dark blue denim jeans"</figcaption>
    </figure>
  </div>
</div>

<!-- === Four videos in one background, same height === -->
<style>
  .video-block {
    background:#f5f6f7;
    padding:18px;
    border-radius:14px;
    box-shadow:0 0 12px rgba(0,0,0,.08);
    margin:20px 0;
  }
  .video-row {
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:16px;  /* 间距比之前小 */
    align-items:start;
  }
  .video-row figure {
    text-align:center;
    margin:0;
  }
  /* 视频样式：统一高度 */
  .video-row video {
    display:block;       /* 避免基线问题 */
    width:100%;
    height:360px;   /* 固定高度，保持一致 */
    object-fit:cover;
    border-radius:10px;
  }
  /* 说明文字 */
  .video-row figcaption {
    margin-top:6px;
    font-size:14px;
    font-style:italic;
    color:#333;
  }
/* 盖掉主题对 figure 的默认外边距，避免第2~4个被往下推 */
.video-row figure { 
  margin: 0 !important;
}
.video-row figure + figure {
  margin-top: 0 !important;
}
</style>

<div class="video-block">
  <div class="video-row">
    <figure>
      <video autoplay loop muted playsinline>
        <source src="static/video/source2.mp4" type="video/mp4">
      </video>
      <figcaption>Source</figcaption>
    </figure>
    <figure>
      <video autoplay loop muted playsinline>
        <source src="static/video/dress2.mp4" type="video/mp4">
      </video>
      <figcaption>"Make her in  a dress"</figcaption>
    </figure>
    <figure>
      <video autoplay loop muted playsinline>
        <source src="static/video/ruffle2.mp4" type="video/mp4">
      </video>
      <figcaption>"Turn her T-shirt into a ruffle dress"</figcaption>
    </figure>
    <figure>
      <video autoplay loop muted playsinline>
        <source src="static/video/lace22.mp4" type="video/mp4">
      </video>
      <figcaption>"Turn her T-shirt into a lace blouse"</figcaption>
    </figure>
  </div>
</div>


