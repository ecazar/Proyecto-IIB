# Proyecto-IIB
# 🛍️ Sistema de Recuperación de Información Multimodal

Sistema completo de búsqueda multimodal aplicado a productos de Amazon que integra retrieval vectorial, re-ranking con cross-encoders y generación aumentada por recuperación (RAG) con capacidades conversacionales.

---

## 🎯 Características

### 🔍 Búsqueda Multimodal
- **📝 Text-to-Product**: Búsqueda de productos mediante consultas textuales
- **🖼️ Image-to-Product**: Búsqueda por similitud visual usando imágenes
- **🔄 Búsqueda Híbrida**: Combinación ponderada de texto e imagen 

### 🎯 Re-ranking Inteligente


### 🤖 RAG (Retrieval-Augmented Generation)
- 💬 Generación de respuestas explicativas con Gemini 2.5 Flash
- ⭐ Justificación basada en reseñas reales de usuarios

### 💭 Búsqueda Conversacional
- 🧠 Memoria de sesión (últimos 2 turnos)
- 🎯 Detección automática de refinamientos
- 🔄 Reescritura inteligente de consultas con LLM
- ⚡ Caché de reescrituras para optimizar rendimiento

---

## 💻 Requisitos

### 🖥️ Software
- 🐍 Python 3.8+
- ☁️ Google Colab (recomendado) o Jupyter Notebook
- 💾 Google Drive (para persistencia de datos)

### 🔧 Hardware
- 🎮 GPU recomendada para encoding (T4 o superior en Colab)
- 💾 Mínimo 12GB RAM
- 📦 5GB de espacio en disco

### 🔑 APIs
- 🌟 Google Gemini API Key

---

## 📦 Instalación

### 1️⃣ Clonar el repositorio

```bash
git clone https://github.com/ecazar/Proyecto-IIB.git
```

### 2️⃣ Configurar credenciales

**🔑 Gemini API:**

```python
# En tu notebook
import os
os.environ["GEMINI_API_KEY"] = "tu-api-key-aqui"
```

---
