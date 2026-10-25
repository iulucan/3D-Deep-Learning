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
