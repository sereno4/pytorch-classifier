# 🖼️ Classificador de Imagens com PyTorch

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-F472B6?logo=gradio&logoColor=white)](https://gradio.app)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD166?logo=huggingface&logoColor=black)](https://huggingface.co)

App interativo que identifica objetos em imagens usando **ResNet18 pré-treinada** no ImageNet.

🔗 **Experimente online:** https://Danielfonseca1212-pytorch-classifier.hf.space

---

## 📸 Demonstração

![Demo](https://i.imgur.com/7ZlB9uP.png)
*(Substitua por um print real do seu app funcionando)*

---

## 🚀 Tecnologias

- **PyTorch** - Framework de deep learning
- **TorchVision** - Modelos pré-treinados (ResNet18)
- **Gradio** - Interface web interativa
- **Hugging Face Spaces** - Deploy em nuvem

---

## 📋 Como funciona

1. **Upload** - Usuário envia uma imagem
2. **Pré-processamento** - Resize, crop e normalização compatível com ImageNet
3. **Inferência** - ResNet18 pré-treinada processa a imagem
4. **Pós-processamento** - Softmax converte logits em probabilidades
5. **Resultado** - Exibição do Top-3 classes com porcentagens

---

## 🛠️ Instalação Local

```bash
# 1. Clonar repositório
git clone https://github.com/Danielfonseca1212/pytorch-classifier.git
cd pytorch-classifier

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Executar app
python app.py
