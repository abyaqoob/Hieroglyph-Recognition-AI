<h1 align="center">Egyptian Hieroglyph Recognition: IGSM-CVV Framework</h1>

<p align="center">
  <i>Automated recognition of ancient Egyptian hieroglyphs optimized for accessibility and high-accuracy classification.</i>
</p>

<hr>

<h3>🚀 Key Features</h3>
<ul>
  <li><b>Hardware Optimized:</b> Specifically tuned for 12GB VRAM environments (NVIDIA T4) using <b>PyTorch Mixed Precision (AMP)</b>.</li>
  <li><b>IGSM Segmentation:</b> A hybrid model combining Local Adaptive Gaussian Thresholding with the <b>Segment Anything Model (SAM)</b>.</li>
  <li><b>CVV Ensemble:</b> A 5-slot Cross-Validation Voting system utilizing a <b>ConvNeXt-tiny</b> backbone.</li>
  <li><b>Curated Dataset:</b> High-fidelity training on 155 Gardiner sign classes with strict stratification.</li>
</ul>

<hr>

<h3>🏗️ Architecture Overview</h3>
<p>The pipeline follows a three-stage process: Preprocessing (IGSM), Feature Extraction, and Ensemble Classification (CVV).</p>

<ol>
  <li>
    <b>Image Generation & Segmentation (IGSM):</b> 
    Resolves low-contrast stone carvings by using Gaussian thresholding as a spatial prompt for SAM.
  </li>
  <li>
    <b>Cross-Validation Voting (CVV):</b> 
    Utilizes a <b>Soft Voting (SV)</b> system with five diverse model weights to reduce error rates for visually similar signs.
  </li>
</ol>

<hr>

<h3>📊 Performance Metrics</h3>
<p>Our ensemble approach demonstrates superior stability across unbalanced classes compared to baseline CNN architectures.</p>

<table width="100%">
  <thead>
    <tr style="background-color: #2d333b;">
      <th align="left">Metric</th>
      <th align="left">Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Global Accuracy</b></td>
      <td>0.958</td>
    </tr>
    <tr>
      <td><b>Balanced Accuracy</b></td>
      <td>0.899</td>
    </tr>
    <tr>
      <td><b>F1 Score</b></td>
      <td>0.898</td>
    </tr>
    <tr>
      <td><b>Gardiner Classes Supported</b></td>
      <td>155</td>
    </tr>
    <tr>
      <td><b>Inference Latency</b></td>
      <td>~45ms / image</td>
    </tr>
  </tbody>
</table>

<hr>

<p align="center">
  <i>Developed for the Artificial Intelligence (CS3002) course project.</i>
</p>
