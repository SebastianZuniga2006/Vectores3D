import re
import streamlit as st # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify, diff, lambdify
from sympy.vector import CoordSys3D
from io import BytesIO

st.set_page_config(page_title="Rotacional y Vectores en 3 dimensiones", layout="wide")

st.title("Operaciones con vectores en 3 dimensiones y cálculo del rotacional de un campo vectorial")

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
v1_input = st.sidebar.text_input("Vector v1 (ej: (1,2,3))", value="(1,2,3)")
v2_input = st.sidebar.text_input("Vector v2 (ej: (2,0,-1))", value="(2,0,-1)")

# === Campo vectorial ===
st.sidebar.subheader("Campo vectorial F(x, y, z)")
Fx_expr = st.sidebar.text_input("Fₓ(x, y, z)", "2x")
Fy_expr = st.sidebar.text_input("Fᵧ(x, y, z)", "-y")
Fz_expr = st.sidebar.text_input("F_z(x, y, z)", "z")

# === Densidad de malla para el campo rotacional ===
densidad = st.sidebar.slider("Densidad del campo (puntos por eje)", 3, 15, 7)

if st.sidebar.button("Calcular"):
    def parse_vector_parenthesis (v_str):
        try:
            v_str = v_str.strip("() ")
            vector = np.array([float(x) for x in v_str.split(",")])
            return vector
        except:
            st.error("❌ Error al interpretar el vector. Usa el formato (1,2,3)")
            return None 
    def parse_vector_rot (v_str):
        try:
            v_str = v_str.replace("^","**").replace("*","").replace(" ","")
            componentes = v_str.split(",")

            if len(componentes) != 3:
                raise ValueError ("Debe ingresar exactamente 3 componentes separadas por comas.")
            
            componentes_finales = []
            for expr in componentes:
                expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
                expr = re.sub(r'([a-zA-Z])([a-zA-Z])', r'\1*\2', expr)
                componentes_finales.append(sympify(expr, evaluate = True))
            return componentes_finales
        
        except:
            st.error("❌ Error al interpretar el vector. Usa el formato: 2x, -y, z")
            return None
    def show_vector (name, v, subscript  = None, use_text = False):
        coords = ',\ '.join(f"{x:.3f}" for x in v)

        if subscript:
            if use_text:
                vector_name = rf"\vec{{{name}_{{\text{{{subscript}}}}}}}"
            else:
                vector_name = rf"\vec{{{name}_{{{subscript}}}}}"
        else:
            vector_name = rf"\vec{{{name}}}"
        
        st.latex(rf"{vector_name} = \left({coords}\right)")
    
    try:
        # --- Conversión de vectores ---
        vector1 = parse_vector_parenthesis(v1_input)    
        vector2 = parse_vector_parenthesis(v2_input)  
        cruz = np.cross(vector1, vector2)

        if np.linalg.norm(vector2) == 0:
            raise ValueError("El vector 2 no puede ser nulo para proyectar.")
        proy = (np.dot(vector1, vector2) / np.linalg.norm(vector2)**2) * vector2

        # --- Campo vectorial simbólico ---
        x, y, z = symbols('x y z')

        vector_expr = f"{Fx_expr},{Fy_expr},{Fz_expr}"
        vector_components = parse_vector_rot(vector_expr)

        if not vector_components:
            st.stop()

        Fx, Fy, Fz = parse_vector_rot(f"{Fx_expr}, {Fy_expr}, {Fz_expr}")

        # --- Rotacional ---
        rot_x = diff(Fz, y) - diff(Fy, z)
        rot_y = -(diff(Fz, x) - diff(Fx, z))
        rot_z = diff(Fy, x) - diff(Fx, y)

        # --- Mostrar resultados ---
        if vector1 is not None and vector2 is not None:
            st.subheader("🧮 Resultados")
            st.write("Vector 1: ")
            show_vector("v",vector1, subscript = "1")
            st.write("Vector 2: ")
            show_vector("v", vector2, subscript= "2")
            st.write("Proyección: ")
            show_vector("v", proy, subscript= "1_{v_2}")
            st.write("Producto cruz: ")
            coords = ',\ '.join(f"{x:.3f}" for x in cruz)
            st.latex(r"\vec{v}_1 \times \vec{v}_2 = \left(" + coords + r"\right)")

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

        dibujar_vector(vector1, 'blue', 'v1')
        dibujar_vector(vector2, 'green', 'v2')
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
        U, V, W = 2*U, 2*V, 2*W 
        ax2.quiver(X, Y, Z, U, V, W, length=0.5, normalize=False, color='purple')
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
        st.download_button("💾 Descargar imagen de vectores", buffer.getvalue(), "vectores.png", mime="image/png", key = "vectores", help = None, on_click = "ignore")

        buffer2 = BytesIO()
        fig2.savefig(buffer2, format="png")
        st.download_button("💾 Descargar imagen del rotacional", buffer2.getvalue(), "rotacional.png", mime="image/png", key = "rotacional", help = None, on_click = "ignore")

    except Exception as e:
        st.error(f"❌ Error: {e}")
