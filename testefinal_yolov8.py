from PIL import Image
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8x-cls.pt")

# Run inference on 'bus.jpg'
results = model("image.jpg", save_txt=True)  # results list

# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    #r.show()

    # Save results to disk
    r.save(filename=f"results{i}.jpg")


# Tratamento dos dados

with open("runs/classify/predict/labels/image.txt") as f:
    content = f.readlines()

content = [x.strip('\n') for x in content]

infogeral = {}
infodetalhe = {}

for f in range(len(content)):
    infogeral[str.title(content[f].split(' ')[1][0:])] = float(content[f].split(' ')[0][0:])
    if infogeral[str.title(content[f].split(' ')[1][0:])] >= 0.70:
        infodetalhe[str.title(content[f].split(' ')[1][0:])] = float(content[f].split(' ')[0][0:])


print("\nDETECÇÕES APROVADAS")
for detect, conf in infodetalhe.items():
     print(f" {detect}: {conf}")

print("\nTODAS AS DETECÇÕES")
for detect, conf in infogeral.items():
     print(f" {detect}: {conf}")


# Mostrar para o usuário a opção semelhando a partir da foto pesquisada

if "Microwave" in infodetalhe:
     image = Image.open("./semelhantes/microwave/microwave.jpg")
     image.show()

if "Notebook" in infodetalhe:
     image = Image.open("./semelhantes/notebook/notebook.jpg")
     image.show()

if "Refrigerator" in infodetalhe:
     image = Image.open("./semelhantes/refrigerator/refrigerator.jpg")
     image.show()

if "Teddy" in infodetalhe:
     image = Image.open("./semelhantes/teddy/teddy.jpg")
     image.show()