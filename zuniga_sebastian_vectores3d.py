import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify, diff, lambdify
from sympy.vector import CoordSys3D
from io import BytesIO

st.set_page_config(page_title="Rotacional y Vectores en 3D", layout="wide")

st.title("Cálculo del Rotacional y Vectores en 3D")

st.markdown("""
Esta aplicación permite:
- Calcular el **producto cruz** entre dos vectores.
- Calcular la **proyección** de un vector sobre otro.
- Ingresar un **campo vectorial F(x,y,z)** y calcular su **rotacional simbólico**.
- Visualizar en 3D:
  - Los vectores: `v1`, `v2`, proyección y producto cruz.
  - El campo del **rotacional** en el espacio 3D.
""")

st.sidebar.header("📥 Ingreso de datos")

# === Entrada de vectores ===
v1_input = st.sidebar.text_input("Vector v1 (ej: 1,2,3)", value="1,2,3")
v2_input = st.sidebar.text_input("Vector v2 (ej: 2,0,-1)", value="2,0,-1")

# === Campo vectorial ===
st.sidebar.subheader("Campo vectorial F(x, y, z)")
Fx_expr = st.sidebar.text_input("Fₓ(x, y, z)", "y*z")
Fy_expr = st.sidebar.text_input("Fᵧ(x, y, z)", "z*x")
Fz_expr = st.sidebar.text_input("F_z(x, y, z)", "x*y")

# === Densidad de malla para el campo rotacional ===
densidad = st.sidebar.slider("Densidad del campo (puntos por eje)", 3, 10, 5)

if st.sidebar.button("Calcular"):
    try:
        # --- Conversión de vectores ---
        v1 = np.array([float(x.strip()) for x in v1_input.split(",")])
        v2 = np.array([float(x.strip()) for x in v2_input.split(",")])
        cruz = np.cross(v1, v2)

        if np.linalg.norm(v2) == 0:
            raise ValueError("El vector v2 no puede ser nulo para proyectar.")
        proy = (np.dot(v1, v2) / np.linalg.norm(v2)**2) * v2

        # --- Campo vectorial simbólico ---
        x, y, z = symbols('x y z')
        Fx = sympify(Fx_expr)
        Fy = sympify(Fy_expr)
        Fz = sympify(Fz_expr)

        # --- Rotacional ---
        rot_x = diff(Fz, y) - diff(Fy, z)
        rot_y = -(diff(Fz, x) - diff(Fx, z))
        rot_z = diff(Fy, x) - diff(Fx, y)

        # --- Mostrar resultados ---
        st.subheader("🧮 Resultados")
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
        st.subheader("📌 Visualización de vectores")
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection='3d')

        def dibujar_vector(v, color, label):
            ax1.quiver(0, 0, 0, v[0], v[1], v[2], color=color, label=label)

        dibujar_vector(v1, 'blue', 'v1')
        dibujar_vector(v2, 'green', 'v2')
        dibujar_vector(cruz, 'red', 'v1 × v2')
        dibujar_vector(proy, 'orange', 'Proy. de v1 sobre v2')

        ax1.set_xlim([-5, 5])
        ax1.set_ylim([-5, 5])
        ax1.set_zlim([-5, 5])
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title("Vectores en 3D")
        ax1.legend()

        st.pyplot(fig1)

        # --- Gráfico del campo del rotacional ---
        st.subheader("🌐 Visualización del campo del rotacional")

        N = CoordSys3D('N')
        fx = lambdify((x, y, z), rot_x, 'numpy')
        fy = lambdify((x, y, z), rot_y, 'numpy')
        fz = lambdify((x, y, z), rot_z, 'numpy')

        dom = np.linspace(-2, 2, densidad)
        X, Y, Z = np.meshgrid(dom, dom, dom)
        U = fx(X, Y, Z)
        V = fy(X, Y, Z)
        W = fz(X, Y, Z)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.quiver(X, Y, Z, U, V, W, length=0.3, normalize=True, color='purple')
        ax2.set_title("Campo vectorial del rotacional")
        ax2.set_xlim([-2, 2])
        ax2.set_ylim([-2, 2])
        ax2.set_zlim([-2, 2])
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")

        st.pyplot(fig2)

        # --- Opción para descargar gráfico como imagen ---
        buffer = BytesIO()
        fig1.savefig(buffer, format="png")
        st.download_button("📥 Descargar imagen de vectores", buffer.getvalue(), "vectores.png", mime="image/png")

        buffer2 = BytesIO()
        fig2.savefig(buffer2, format="png")
        st.download_button("📥 Descargar imagen del rotacional", buffer2.getvalue(), "rotacional.png", mime="image/png")

    except Exception as e:
        st.error(f"❌ Error: {e}")
