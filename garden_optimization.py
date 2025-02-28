import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linprog, minimize_scalar

# Page Configuration
st.set_page_config(page_title="Optimization Visualizer 🚀", layout="wide")

# Sidebar: Algorithm Selection
st.sidebar.title("Select Algorithm 🧮")
algorithm = st.sidebar.selectbox(
    "Choose an Optimization Algorithm",
    ["Fibonacci Search 🔍", "Golden Section Search 🌟", "Gradient Descent 📉", "Linear Programming 📊"]
)

# Sidebar: User Inputs
st.sidebar.header("Input Parameters 🛠️")

if algorithm in ["Fibonacci Search 🔍", "Golden Section Search 🌟"]:
    function_str = st.sidebar.text_input("Enter Function (e.g., 'x**2 + 3*x + 2') 📝", "x**2 + 3*x + 2")
    search_range = st.sidebar.slider("Search Range 📏", -10.0, 10.0, (-5.0, 5.0))
    num_iterations = st.sidebar.number_input("Number of Iterations 🔢", 1, 100, 10)

elif algorithm == "Gradient Descent 📉":
    function_str = st.sidebar.text_input("Enter Function (e.g., 'x**2 + 3*x + 2') 📝", "x**2 + 3*x + 2")
    learning_rate = st.sidebar.number_input("Learning Rate 🔄", 0.001, 1.0, 0.1)
    num_iterations = st.sidebar.number_input("Number of Iterations 🔢", 1, 1000, 50)
    # Define a default search range for Gradient Descent
    search_range = st.sidebar.slider("Search Range (Optional) 📏", -10.0, 10.0, (-5.0, 5.0), key="gd_search_range")

elif algorithm == "Linear Programming 📊":
    st.sidebar.subheader("Objective Function 🎯")
    obj_coeffs = st.sidebar.text_input("Objective Coefficients (comma-separated, e.g., '2, 3') 📝", "2, 3")
    obj_coeffs = np.array([float(c.strip()) for c in obj_coeffs.split(",")])

    st.sidebar.subheader("Inequality Constraints ⚖️")
    A_ub_input = st.sidebar.text_area("Constraint Matrix (rows separated by ';', e.g., '1, 1; -1, 2') 📝", "1, 1; -1, 2")
    A_ub = np.array([[float(c.strip()) for c in row.split(",")] for row in A_ub_input.split(";")])
    b_ub_input = st.sidebar.text_input("Constraint RHS (comma-separated, e.g., '4, 2') 📝", "4, 2")
    b_ub = np.array([float(b.strip()) for b in b_ub_input.split(",")])

    st.sidebar.subheader("Variable Bounds 📏")
    x_bounds = st.sidebar.slider("Bounds for x 📏", -10.0, 10.0, (-5.0, 5.0))
    y_bounds = st.sidebar.slider("Bounds for y 📏", -10.0, 10.0, (-5.0, 5.0))

# Main Panel: Visualization
st.title("Optimization Visualizer 🚀")
st.markdown("""
This app allows you to visualize and solve optimization problems using various algorithms. 🧠✨
""")

# Function to evaluate user-defined function
def eval_function(func_str, x):
    return eval(func_str, {"np": np, "x": x})

# Algorithm Execution
if st.sidebar.button("Run Algorithm 🏃‍♂️"):
    if algorithm == "Fibonacci Search 🔍":
        with st.spinner("Running Fibonacci Search... 🔍"):
            def fibonacci_search(f, a, b, n):
                fib = [0, 1]
                for i in range(2, n + 2):
                    fib.append(fib[-1] + fib[-2])
                x1 = a + (fib[n - 2] / fib[n]) * (b - a)
                x2 = a + (fib[n - 1] / fib[n]) * (b - a)
                history = []
                for _ in range(n):
                    history.append((a, b, x1, x2))
                    if f(x1) < f(x2):
                        b = x2
                        x2 = x1
                        x1 = a + (fib[n - _ - 2] / fib[n - _]) * (b - a)
                    else:
                        a = x1
                        x1 = x2
                        x2 = a + (fib[n - _ - 1] / fib[n - _]) * (b - a)
                return (a + b) / 2, history

            f = lambda x: eval_function(function_str, x)
            result, history = fibonacci_search(f, search_range[0], search_range[1], num_iterations)
            x_vals = np.linspace(search_range[0], search_range[1], 500)
            y_vals = f(x_vals)

            # Plot results
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="Function", line=dict(color="cyan")))
            fig.add_trace(go.Scatter(x=[result], y=[f(result)], mode="markers", name="Minimum", marker=dict(color="yellow", size=10)))
            st.plotly_chart(fig)

            # Display details
            st.subheader("Optimization Details 📋")
            st.write(f"**Function:** {function_str}")
            st.write(f"**Search Range:** {search_range}")
            st.write(f"**Number of Iterations:** {num_iterations}")
            st.write(f"**Optimal Point:** x = {result:.4f}, f(x) = {f(result):.4f}")
            st.write("**Iteration History:**")
            for i, (a, b, x1, x2) in enumerate(history):
                st.write(f"Iteration {i+1}: a={a:.4f}, b={b:.4f}, x1={x1:.4f}, x2={x2:.4f}")

    elif algorithm == "Golden Section Search 🌟":
        with st.spinner("Running Golden Section Search... 🌟"):
            f = lambda x: eval_function(function_str, x)
            res = minimize_scalar(f, bracket=search_range, method='golden')
            x_vals = np.linspace(search_range[0], search_range[1], 500)
            y_vals = f(x_vals)

            # Plot results
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="Function", line=dict(color="cyan")))
            fig.add_trace(go.Scatter(x=[res.x], y=[res.fun], mode="markers", name="Minimum", marker=dict(color="yellow", size=10)))
            st.plotly_chart(fig)

            # Display details
            st.subheader("Optimization Details 📋")
            st.write(f"**Function:** {function_str}")
            st.write(f"**Search Range:** {search_range}")
            st.write(f"**Optimal Point:** x = {res.x:.4f}, f(x) = {res.fun:.4f}")

    elif algorithm == "Gradient Descent 📉":
        with st.spinner("Running Gradient Descent... 📉"):
            f = lambda x: eval_function(function_str, x)
            df = lambda x: (f(x + 1e-6) - f(x)) / 1e-6  # Numerical derivative
            x = np.random.uniform(search_range[0], search_range[1])  # Use search_range defined above
            x_history, y_history = [], []

            for i in range(num_iterations):
                grad = df(x)
                x -= learning_rate * grad
                x_history.append(x)
                y_history.append(f(x))

            x_vals = np.linspace(search_range[0], search_range[1], 500)
            y_vals = f(x_vals)

            # Plot results
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="Function", line=dict(color="cyan")))
            fig.add_trace(go.Scatter(x=x_history, y=y_history, mode="markers+lines", name="Gradient Descent Path", marker=dict(color="yellow", size=5)))
            st.plotly_chart(fig)

            # Display details
            st.subheader("Optimization Details 📋")
            st.write(f"**Function:** {function_str}")
            st.write(f"**Learning Rate:** {learning_rate}")
            st.write(f"**Number of Iterations:** {num_iterations}")
            st.write(f"**Final Point:** x = {x_history[-1]:.4f}, f(x) = {y_history[-1]:.4f}")
            st.write("**Iteration History:**")
            for i, (xi, yi) in enumerate(zip(x_history, y_history)):
                st.write(f"Iteration {i+1}: x = {xi:.4f}, f(x) = {yi:.4f}")

    elif algorithm == "Linear Programming 📊":
        with st.spinner("Solving Linear Programming Problem... 📊"):
            bounds = [x_bounds, y_bounds]
            result = linprog(c=obj_coeffs, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

            # Display Results
            st.subheader("Problem Setup 📝")
            st.write(f"**Objective Function:** Minimize {obj_coeffs[0]}*x + {obj_coeffs[1]}*y")
            st.write("**Constraints:**")
            for i, row in enumerate(A_ub):
                st.write(f"{row[0]}*x + {row[1]}*y <= {b_ub[i]}")
            st.write(f"**Bounds:** x ∈ {x_bounds}, y ∈ {y_bounds}")

            st.subheader("Solution Details 📋")
            if result.success:
                st.write(f"**Optimal Solution Found:** x = {result.x[0]:.2f}, y = {result.x[1]:.2f}")
                st.write(f"**Optimal Objective Value:** {result.fun:.2f}")
            else:
                st.error("No feasible solution found. ❌")

            # Visualization
            st.subheader("Feasible Region and Optimal Point 📈")
            fig = go.Figure()

            # Dynamically calculate plot limits based on constraints and bounds
            x_min, x_max = min(x_bounds[0], -5), max(x_bounds[1], 5)
            y_min, y_max = min(y_bounds[0], -5), max(y_bounds[1], 5)

            # Plot constraints
            x_vals = np.linspace(x_min, x_max, 400)
            for i, row in enumerate(A_ub):
                if row[1] != 0:  # Avoid division by zero
                    y_vals = (b_ub[i] - row[0] * x_vals) / row[1]
                    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name=f"{row[0]}*x + {row[1]}*y <= {b_ub[i]}", line=dict(color="cyan")))

            # Highlight feasible region
            x_fill = np.linspace(x_min, x_max, 400)
            y_fill = np.linspace(y_min, y_max, 400)
            X, Y = np.meshgrid(x_fill, y_fill)
            Z = np.zeros_like(X)
            for i, row in enumerate(A_ub):
                Z += (row[0] * X + row[1] * Y <= b_ub[i]).astype(int)
            mask = Z == len(A_ub)
            fig.add_trace(go.Contour(x=x_fill, y=y_fill, z=mask.astype(float), showscale=False, opacity=0.3, colorscale=[[0, "gray"], [1, "blue"]]))

            # Plot optimal point
            if result.success:
                fig.add_trace(go.Scatter(x=[result.x[0]], y=[result.x[1]], mode="markers", name="Optimal Point", marker=dict(color="yellow", size=10)))

            # Customize layout
            fig.update_layout(
                title="Feasible Region and Optimal Point",
                xaxis_title="x",
                yaxis_title="y",
                showlegend=True,
                template="plotly_dark"
            )
            st.plotly_chart(fig)

# Additional UI Enhancements
st.sidebar.markdown("---")
st.sidebar.info("Use tooltips and instructions here. ℹ️")