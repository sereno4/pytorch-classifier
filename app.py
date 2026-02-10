import torch
import torchvision
from torchvision import transforms
from PIL import Image
import gradio as gr

# Carregar modelo ResNet18 pré-treinado
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# Pré-processamento padrão ImageNet
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Carregar labels do ImageNet
try:
    labels = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
except:
    labels = [f"Classe {i}" for i in range(1000)]

def predict(image):
    """
    Classifica uma imagem usando ResNet18 pré-treinada.
    
    Args:
        image: Imagem PIL carregada pelo usuário
        
    Returns:
        str: Top-3 previsões formatadas
    """
    # Converter para RGB se necessário
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Pré-processamento
    img_t = preprocess(image).unsqueeze(0)
    
    # Inferência (sem gradientes)
    with torch.no_grad():
        output = model(img_t)
    
    # Converter logits → probabilidades
    probs = torch.nn.functional.softmax(output[0], dim=0)
    
    # Top-3 classes
    top3_prob, top3_catid = torch.topk(probs, 3)
    
    # Formatar resultado
    result = "🏆 Top 3 previsões:\n\n"
    for i in range(3):
        prob = float(top3_prob[i]) * 100
        label = labels[int(top3_catid[i])]
        result += f"{i+1}. {label:<25} → {prob:.1f}%\n"
    
    return result

# Criar interface Gradio
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="📸 Envie uma imagem"),
    outputs=gr.Textbox(label="🤖 Resultado da IA"),
    title="🖼️ Classificador PyTorch",
    description="Carregue qualquer imagem e veja a IA identificar objetos em tempo real!",
    theme=gr.themes.Soft()
)

# Executar app
if __name__ == "__main__":
    iface.launch()
