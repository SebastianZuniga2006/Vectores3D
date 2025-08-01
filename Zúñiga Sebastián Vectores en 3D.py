import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify, diff, pprint, init_printing
from sympy.vector import CoordSys3D
from io import BytesIO

init_printing()

st.set_page_config(page_title="Rotacional y Vectores 3D", layout="wide")

st.title("🌪️ Cálculo del Rotacional y Vectores en 3D")
st.markdown("""
Esta aplicación permite:
- Calcular el **producto cruz** entre dos vectores.
- Calcular la **proyección** de un vector sobre otro.
- Ingresar un **campo vectorial F(x,y,z)** y calcular su **rotacional**.
- Visualizar gráficos 3D interactivos.
""")

st.sidebar.header("📥 Ingreso de datos")

# --- Entradas de vectores ---
v1_input = st.sidebar.text_input("Vector v1 (ej: 1,2,3)", value="1,2,3")
v2_input = st.sidebar.text_input("Vector v2 (ej: 2,0,-1)", value="2,0,-1")

# --- Campo vectorial ---
st.sidebar.subheader("Campo vectorial F(x, y, z)")
Fx_expr = st.sidebar.text_input("Fₓ(x, y, z)", "y*z")
Fy_expr = st.sidebar.text_input("Fᵧ(x, y, z)", "z*x")
Fz_expr = st.sidebar.text_input("F_z(x, y, z)", "x*y")

st.sidebar.markdown("⬇️ Presiona el botón para realizar los cálculos")
calcular = st.sidebar.button("Calcular")

if calcular:
    try:
        # Convertir vectores a NumPy
        v1 = np.array([float(x.strip()) for x in v1_input.split(",")])
        v2 = np.array([float(x.strip()) for x in v2_input.split(",")])

        # Cálculos vectoriales
        cruz = np.cross(v1, v2)
        if np.linalg.norm(v2) == 0:
            raise ValueError("El vector v2 no puede ser nulo para proyectar.")
        proy = (np.dot(v1, v2) / np.linalg.norm(v2)**2) * v2

        # Campo vectorial simbólico
        x, y, z = symbols('x y z')
        Fx = sympify(Fx_expr)
        Fy = sympify(Fy_expr)
        Fz = sympify(Fz_expr)

        # Rotacional
        rot_x = diff(Fz, y) - diff(Fy, z)
        rot_y = -(diff(Fz, x) - diff(Fx, z))
        rot_z = diff(Fy, x) - diff(Fx, y)

        # Mostrar resultados
        st.subheader("🧮 Resultados simbólicos y numéricos")
        st.write(f"**v1:** {v1}")
        st.write(f"**v2:** {v2}")
        st.write(f"**Producto cruz v1 × v2:** {cruz}")
        st.write(f"**Proyección de v1 sobre v2:** {proy}")
        st.markdown(f"""
        **Campo vectorial**:  
        $\\vec{{F}}(x, y, z) = \\langle {Fx_expr},\ {Fy_expr},\ {Fz_expr} \\rangle$

        **Rotacional**:  
        $\\nabla \\times \\vec{{F}} = \\langle {rot_x},\ {rot_y},\ {rot_z} \\rangle$
        """)

        # --- Gráfico de vectores ---
        st.subheader("📊 Visualización en 3D")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        def dibujar_vector(v, color, label):
            ax.quiver(0, 0, 0, v[0], v[1], v[2], color=color, label=label)

        dibujar_vector(v1, 'blue', 'v1')
        dibujar_vector(v2, 'green', 'v2')
        dibujar_vector(cruz, 'red', 'v1 × v2')
        dibujar_vector(proy, 'orange', 'Proy. de v1 sobre v2')

        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_zlim([-5, 5])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        ax.set_title("Vectores en el espacio 3D")

        st.pyplot(fig)

        # Descargar resultados como archivo
        buffer = BytesIO()
        fig.savefig(buffer, format="png")
        st.download_button(
            label="📥 Descargar imagen",
            data=buffer.getvalue(),
            file_name="grafico_vectores.png",
            mime="image/png"
        )

    except Exception as e:
        st.error(f"❌ Error durante el cálculo: {e}")
