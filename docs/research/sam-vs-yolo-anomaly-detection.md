# Solar Panel Anomaly Detection: SAM vs. YOLO

> **Status:** Research proposal / evaluation plan — no experimental performance claims are made here.  
> **Last reviewed:** 2026-10-25  
> **Scope:** Solar photovoltaic (PV) module inspection using RGB and thermal imagery, including drone-based surveys.

## 1. Research question

**Could Segment Anything (SAM) and YOLO improve solar panel anomaly detection, particularly for hotspots, cracks, damaged cells, and other visible or thermal defects? Should they be used separately or together?**

### Short answer

Both are promising, but serve different purposes:

- **YOLO** can be fine-tuned to **detect and classify known defects** using annotated images. Detection models output bounding boxes; segmentation variants can output pixel masks.
- **SAM** is primarily a **promptable segmentation** family. It can isolate panels or candidate defect regions but does not, by itself, reliably determine whether a panel has an electrical or thermal fault.
- **SAM 3/3.1** can also detect and segment instances of text- or exemplar-described visual concepts. This is not equivalent to validated defect diagnosis, particularly for subtle thermal anomalies.
- A **hybrid approach** can use YOLO for detection and SAM for mask refinement, but its benefit must be tested against a simpler YOLO-seg baseline.

## 2. Proposed applications

| Inspection task | YOLO | SAM family | Additional considerations |
|---|---|---|---|
| Identify individual PV modules | Train a panel detector/segmenter | Prompted panel masks, or concept-based masks with SAM 3 | Perspective, reflections, overlapping panel boundaries |
| Detect visible cracks or physical damage | Train on labeled defects | Refine boundaries of visible suspected defects | Hairline cracks may require close-up or electroluminescence images |
| Detect thermal hotspots | Train on labeled thermal patterns | Segment thermal regions or panel regions | Use calibrated radiometric temperature data where available |
| Detect unknown/rare defects | Limited without representative labels | Masks alone do not identify novelty | Compare with dedicated unsupervised anomaly methods |
| Process drone video | Fast object detection/tracking workflows | SAM 2+ supports video mask propagation | Assess motion, camera resolution, frame rate, compute |
| Produce pixel-accurate defect areas | YOLO segmentation variants | Strong prompt-based mask refinement | Evaluate mask IoU/Dice and annotation effort |

**Important:** Image appearance alone does not conclusively establish the root cause of a PV fault. Environmental conditions, sensor calibration, electrical tests, and expert review may be necessary.

## 3. SAM versions

| Model | Approximate introduction | Main capability | Solar PV inspection potential | Limitation |
|---|---|---|---|---|
| **SAM (SAM 1)** | 2023 | Promptable image segmentation (points/boxes) | Create panel masks and support labeling | No native video tracking or defect classification |
| **SAM 2** | 2024 | Promptable image/video segmentation with memory | Segment panels and propagate masks through inspection video | Not a dedicated anomaly detector |
| **SAM 2.1** | 2024 | Improved SAM 2 checkpoints and training tools | Practical baseline for panel masking and video workflows | Checkpoint size vs. speed trade-off |
| **SAM 3** | 2025 | Text/exemplar-guided concept detection, segmentation and tracking | Explore open-vocabulary panel identification | Domain-specific defects still require validation |
| **SAM 3.1** | 2026 | More efficient SAM 3 video tracking via object multiplexing | Evaluate multiple modules across video frames | Hardware and checkpoint access requirements |

### SAM 2.1 checkpoint comparison

Official SAM 2.1 benchmark figures (A100 GPU; **not** PV-specific results):

| Checkpoint | Parameters | Reported FPS | Practical starting point |
|---|---:|---:|---|
| `sam2.1_hiera_tiny` | 38.9M | 91.2 | Speed-oriented baseline |
| `sam2.1_hiera_small` | 46.0M | 84.8 | Lightweight alternative |
| `sam2.1_hiera_base_plus` | 80.8M | 64.1 | Balance of speed and capability |
| `sam2.1_hiera_large` | 224.4M | 39.5 | Larger-model comparison |

These FPS figures come from Meta's video-segmentation benchmarks and **must not** be used as expected drone inspection throughput.

## 4. YOLO versions

YOLO is a model **family**, not a single continuous implementation. Versions have different authors, architectures, licenses, supported tasks, and benchmark conditions; higher version numbers do not guarantee better performance on solar-panel images.

| Model family | Year | Notable characteristic | Role in proposed experiments |
|---|---:|---|---|
| **YOLOv5** | 2020 | Widely used PyTorch detection ecosystem | Legacy/reproducibility baseline |
| **YOLOv6** | 2022 | Industrial-efficiency-focused detector | Optional detector comparison |
| **YOLOv7** | 2022 | Real-time detection architecture | Optional research baseline |
| **YOLOv8** | 2023 | Accessible detection and segmentation workflows | Established detection/segmentation baseline |
| **YOLOv9** | 2024 | Programmable gradient information / GELAN | Architecture comparison |
| **YOLOv10** | 2024 | NMS-free end-to-end detection design | Latency-focused comparison |
| **YOLO11** | 2024 | Mature Ultralytics multi-task family | **Recommended stable baseline** |
| **YOLO12** | 2025 | Attention-centric architecture | Optional research comparison |
| **YOLO26** | 2026 | Ultralytics family emphasizing efficient deployment | **Recommended newer baseline** |

**As of 2026-10-25:** Ultralytics lists YOLO26 as its latest released family and YOLO27 as upcoming, not released. The table is a selective list, not a complete history of every YOLO implementation.

### Model size choices

When available in the selected family, test **nano/small** models first for edge or drone workflows, and compare with a **medium/large** model for accuracy. Exact checkpoint names and task support should be checked in the official documentation for each family.

Example checkpoint names:

- Detection: `yolo11n.pt`, `yolo26n.pt`
- Instance segmentation: `yolo11n-seg.pt`, `yolo26n-seg.pt`

The examples are **pretrained general-purpose weights**. They do not recognize solar faults out of the box without suitable task-specific training or validation.

## 5. SAM vs. YOLO: practical comparison

| Criterion | YOLO detection | YOLO segmentation | SAM 2/2.1 | SAM 3/3.1 |
|---|---|---|---|---|
| Main output | Boxes + class scores | Boxes + masks + classes | Prompted object masks | Concept/visual-prompt masks, detection/tracking |
| Defect classification | **Yes, after task-specific training** | **Yes, after task-specific training** | No dedicated defect classifier | Concept matching; not validated fault diagnosis |
| Requires labeled defect examples | Usually yes | Usually yes | Not for prompted segmentation; labels useful for fine-tuning | Prompts possible; domain validation still necessary |
| Annotation support | Boxes | Pixel masks | Strong interactive mask assistance | Text/exemplar-assisted mask creation |
| Previously unseen fault types | Limited | Limited | Can segment a prompted region but not identify a fault | May find visually describable concepts; no anomaly guarantee |
| Speed/resource profile | Often attractive for deployment | Additional mask computation | Depends on checkpoint, prompts and hardware | Depends on prompts, scale and hardware |
| Best initial use | Detect known faults | Detect + outline known faults | Segment individual panels/refine masks | Explore concept-driven panel segmentation |

## 6. Suggested experimental pipelines

### Pipeline A — Supervised YOLO baseline

```text
RGB / thermal inspection images
             |
             v
      Quality control
             |
             v
  Annotated defect dataset
             |
             v
  YOLO11 / YOLO26 training
             |
             v
 Defect class + bounding box
             |
             v
     Evaluation / review
```

Start here if there is a sufficiently labeled defect dataset.

### Pipeline B — YOLO segmentation baseline

```text
RGB / thermal images
          |
          v
  Defect mask annotations
          |
          v
 YOLO11-seg / YOLO26-seg
          |
          v
 Defect class + pixel mask
```

Use this to check whether SAM refinement is really needed.

### Pipeline C — YOLO + SAM 2.1 hybrid

```text
RGB / thermal images
          |
          v
 YOLO defect/panel detection
          |
          v
   Candidate bounding boxes
          |
          v
  SAM 2.1 mask refinement
          |
          v
  Refined region boundaries
          |
          v
 Temperature / defect analysis
          |
          v
  Expert validation + report
```

Bounding boxes can be passed to SAM as prompts. Measure whether the hybrid improves masks enough to justify inference cost and integration complexity. **Segmenting a hotspot-looking region does not validate that it is an electrical fault.**

### Pipeline D — Novel anomaly discovery (separate baseline)

```text
Mostly healthy panel images
            |
            v
 Panel extraction (SAM/YOLO)
            |
            v
 PatchCore / EfficientAD, etc.
            |
            v
    Anomaly score / map
            |
            v
   Expert review + labeling
```

This explores anomalies not exhaustively represented by defect classes. It should be evaluated separately from supervised YOLO training.

## 7. Dataset and evaluation design

### Data requirements

- RGB images and/or thermal images, with sensor type and acquisition conditions recorded.
- Clear distinction between **false-color thermal screenshots** and **radiometric temperature measurements**.
- Labels for panel locations, known defect classes, and (if available) pixel-level defect masks.
- Diverse panel types, lighting, weather, camera angles, altitudes, and healthy controls.
- Ground-truth review by domain experts; electrical measurements where appropriate.
- Train/validation/test splits **by site, inspection flight, or module group**, not random adjacent frames, to avoid leakage.

### Suggested defect taxonomy (adapt to available ground truth)

- Thermal hotspots / abnormal heating patterns
- Visible cracks or physical damage
- Soiling, shading, vegetation, and occlusion (potential confounders)
- Cell/string-level abnormal patterns where labels and image resolution support them
- Unknown or uncertain anomaly (review required)

### Metrics

| Evaluation target | Measures |
|---|---|
| Known-defect detection | Precision, recall, F1, mAP@50, mAP@50:95 |
| Defect segmentation | Mask IoU, Dice, mask mAP |
| Novel-anomaly ranking | AUROC, AUPRC, false-positive rate at fixed recall |
| Operational usefulness | Missed critical faults, false alarms per inspected panel, annotation time |
| Deployment | Latency, throughput, GPU/CPU memory, energy usage |
| Robustness | Per-site, per-camera, per-weather and per-defect breakdown |

**Priority:** Recall for consequential faults, while controlling false positives that lead to unnecessary manual inspections.

### Benchmark matrix

| Experiment | Model | Task | Status |
|---|---|---|---|
| E1 | YOLO11n | Defect detection | Not started |
| E2 | YOLO26n | Defect detection | Not started |
| E3 | YOLO11n-seg | Defect instance segmentation | Not started |
| E4 | YOLO26n-seg | Defect instance segmentation | Not started |
| E5 | SAM 2.1 tiny / base+ | Panel segmentation; annotation support | Not started |
| E6 | YOLO + SAM 2.1 | Detection + mask refinement | Not started |
| E7 | SAM 3 / 3.1 | Concept-guided panel segmentation | Not started |
| E8 | PatchCore or EfficientAD | Novel anomaly baseline | Not started |

## 8. Minimal training example (YOLO)

> **Illustrative commands only.** They require a prepared dataset and a compatible environment; they have not been executed for this project.

```bash
pip install ultralytics

# Check package and runtime information
python -c 'import ultralytics; print(ultralytics.__version__)'

# Train a detection baseline using your own dataset definition
yolo detect train model=yolo11n.pt data=solar_defects.yaml epochs=100 imgsz=640

# Validate using the configured split
yolo detect val model=runs/detect/train/weights/best.pt data=solar_defects.yaml
```

Example `solar_defects.yaml` (paths and class list must be adapted):

```yaml
path: /path/to/solar-dataset
train: images/train
val: images/val
test: images/test
names:
  0: hotspot
  1: visible_damage
  2: soiling
```

For segmentation, use mask annotations and a `*-seg.pt` checkpoint, then the `yolo segment` task rather than `yolo detect`.

## 9. Recommended starting strategy

1. **Establish the data and ground truth**: clarify whether the target is visible damage, thermal patterns, or confirmed electrical faults.
2. **Start with YOLO11n and YOLO26n** as supervised detection baselines.
3. **Compare YOLO-seg** if pixel-level boundaries are required.
4. **Test SAM 2.1** for panel masks, annotation assistance, or refinement of YOLO boxes.
5. **Explore SAM 3/3.1** for open-vocabulary/concept-guided panel identification, without assuming zero-shot fault diagnosis.
6. **Include a dedicated anomaly detector** if rare or unknown faults are part of the goal.
7. **Report accuracy and cost together** on held-out sites/flights before selecting a deployment pipeline.

### Open research questions

- Does a YOLO + SAM pipeline improve defect localization compared with YOLO-seg alone?
- How much annotated solar-specific data is needed to fine-tune YOLO effectively?
- Can SAM 3's text prompting reliably identify panels under different drone-camera viewpoints?
- Does radiometric thermal information improve performance beyond RGB or false-color thermal imagery?
- Which model yields the fewest missed critical faults at an acceptable false-alarm rate?
- Can inference run on the actual inspection hardware within operational limits?

## 10. References and official repositories

1. [Meta: Segment Anything (SAM 1)](https://github.com/facebookresearch/segment-anything)
2. [Meta: SAM 2 / SAM 2.1](https://github.com/facebookresearch/sam2)
3. [Meta: SAM 3](https://github.com/facebookresearch/sam3)
4. [Meta: SAM 3.1 announcement](https://ai.meta.com/blog/segment-anything-model-3/)
5. [Ultralytics: YOLO model documentation](https://docs.ultralytics.com/models/)
6. [Ultralytics: source repository](https://github.com/ultralytics/ultralytics)
7. [Ultralytics: model family history](https://www.ultralytics.com/yolo)

---

*This page describes proposed methods, not validated results for a particular PV dataset. Confirm model licensing, checkpoint access, hardware requirements, and sensor/inspection constraints before production use.*
