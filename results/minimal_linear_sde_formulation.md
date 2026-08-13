# Formulación de una SDE lineal mínima en PCA space

Este documento conecta los modelos gaussianos discretos ya evaluados en PCA
space con una formulación continua mínima tipo SDE. No introduce experimentos
nuevos, no genera secuencias y no afirma que la SDE final de la tesis esté
validada. La formulación debe leerse como un puente metodológico entre los
resultados dinámicos retrospectivos y una futura SDE con términos biológicos
explícitos.

## 1. Estado reducido

Sea `z_t` la representación latente de AntigenLM para una cepa o para un
agregado mensual. Se define el estado reducido:

```math
x_t = \mathrm{PCA}_d(z_t),
\qquad d \in \{3,4,5\}.
```

En los experimentos dinámicos actuales, `x_t` representa el centroide mensual
de un subtipo en PCA space, no una cepa individual.

PCA se usa como espacio reducido preliminar por tres razones:

- 3 componentes principales explican aproximadamente 90% de la varianza global.
- 4 componentes principales explican aproximadamente 95% de la varianza global.
- TwoNN, tras deduplicación HA+NA, sugiere una dimensión intrínseca baja,
  aproximadamente en el rango 3.5-4.8 bajo las configuraciones principales.

Por tanto, PCA space funciona como una primera aproximación controlada para
estudiar dinámica latente antes de intentar una SDE en el espacio completo
`R^384`.

## 2. Modelo random walk gaussiano

El modelo discreto evaluado es:

```math
x_{t+1} = x_t + \mu_\Delta + \varepsilon_t,
\qquad
\varepsilon_t \sim \mathcal{N}(0,Q).
```

Equivalente en incrementos:

```math
\Delta x_t = x_{t+1} - x_t = \mu_\Delta + \varepsilon_t.
```

La SDE continua asociada, en una escala temporal mensual idealizada, es:

```math
dx_t = \mu\,dt + \Sigma\,dW_t,
```

donde `mu` es un drift constante y:

```math
Q \approx \Sigma \Sigma^\top \Delta t.
```

Si se toma `Delta t = 1` mes, entonces:

```math
Q \approx \Sigma \Sigma^\top.
```

Interpretación:

- Este modelo es adecuado como baseline mínimo.
- Es especialmente razonable para H1N1 si el drift estimado es débil y
  persistence/random walk son difíciles de superar.
- No incluye presión inmune explícita ni dirección evolutiva dependiente del
  estado.

## 3. Modelo lineal VAR(1)

El modelo discreto VAR(1) evaluado es:

```math
x_{t+1} = A x_t + b + \varepsilon_t,
\qquad
\varepsilon_t \sim \mathcal{N}(0,Q).
```

Restando `x_t` en ambos lados:

```math
x_{t+1} - x_t = (A-I)x_t + b + \varepsilon_t.
```

Esto sugiere la SDE lineal asociada:

```math
dx_t = \left[(A-I)x_t + b\right]dt + \Sigma dW_t.
```

Esta es una forma de drift lineal, similar a un proceso de Ornstein-Uhlenbeck
generalizado, aunque no necesariamente centrado ni diagonalizable en una base
biológicamente interpretable.

Interpretación:

- Si `A` está cerca de `I`, el modelo se aproxima a un random walk con drift
  débil.
- Si `(A-I)x_t + b` mejora la predicción frente a persistence, hay evidencia de
  drift lineal dependiente del estado.
- En los resultados actuales, H3N2 muestra señal más clara de este tipo que
  H1N1.

## 4. Modelo VAR(2)

El modelo discreto VAR(2) evaluado es:

```math
x_{t+1} = A_1 x_t + A_2 x_{t-1} + b + \varepsilon_t,
\qquad
\varepsilon_t \sim \mathcal{N}(0,Q).
```

Hay dos maneras prudentes de conectarlo con dinámica continua.

### Opción A: modelo autoregresivo discreto

La primera opción es tratar VAR(2) como un modelo discreto mensual, útil para
predicción one-step y evaluación probabilística, sin forzar una interpretación
continua directa.

Esta lectura es conservadora y evita sobreinterpretar `A_2` como un término
biológico.

### Opción B: estado aumentado Markoviano

La segunda opción es aumentar el estado:

```math
y_t =
\begin{bmatrix}
x_t \\
x_{t-1}
\end{bmatrix}.
```

Entonces:

```math
y_{t+1}
=
\begin{bmatrix}
x_{t+1} \\
x_t
\end{bmatrix}
=
\begin{bmatrix}
A_1 & A_2 \\
I & 0
\end{bmatrix}
y_t
+
\begin{bmatrix}
b \\
0
\end{bmatrix}
+
\begin{bmatrix}
\varepsilon_t \\
0
\end{bmatrix}.
```

Así se obtiene una dinámica Markoviana de primer orden en estado aumentado:

```math
y_{t+1} = B y_t + c + \eta_t.
```

Formalmente, una SDE lineal aproximada en estado aumentado sería:

```math
dy_t = \left[(B-I)y_t + c\right]dt + \tilde{\Sigma} dW_t.
```

Interpretación:

- Esta formulación puede capturar inercia o dependencia de incrementos previos.
- En H3N2, VAR(2) fue el modelo más fuerte por RMSE en la corrida principal y
  competitivo en robustez.
- Debe tratarse como una SDE lineal aumentada mínima, no como un modelo de
  selección inmune.

## 5. Estimación de difusión

Para cualquier modelo con media predictiva `x_hat[t+1]`, los residuos son:

```math
\varepsilon_t = x_{t+1} - \hat{x}_{t+1}.
```

La covarianza residual se estima como:

```math
Q = \operatorname{Cov}(\varepsilon_t).
```

Para evitar singularidad o mala condición numérica, se usa regularización:

```math
Q_\lambda = Q + \lambda I.
```

En los experimentos actuales también se exploraron:

- covarianza completa;
- covarianza diagonal;
- inflación de covarianza para calibración.

Una matriz de difusión compatible puede obtenerse mediante Cholesky:

```math
Q_\lambda = LL^\top,
\qquad
\Sigma = L
```

si `Q_lambda` es definida positiva. Alternativamente, puede usarse una
descomposición espectral:

```math
Q_\lambda = U \Lambda U^\top,
\qquad
\Sigma = U \Lambda^{1/2}.
```

Si se interpreta la escala temporal como mensual y `Delta t = 1`, entonces
`Q_lambda` aproxima directamente `Sigma Sigma^T`.

## 6. Conexión con resultados

La lectura actual de los resultados dinámicos es subtipo-dependiente:

- H1N1: persistence y gaussian random walk son baselines fuertes. En el cache
  principal, persistence obtiene el mejor RMSE en `d=5` (`RMSE=0.0695`), y los
  modelos con drift lineal empeoran el error puntual. En el cache seed 7,
  VAR(2) mejora modestamente, por lo que la conclusión prudente es drift débil
  o sensible al muestreo, no ausencia absoluta de drift.
- H3N2: los modelos con drift lineal son más consistentes. En rolling-origin
  con seed 42, VAR(2) en `d=5` obtiene `RMSE=0.0670` y mejora aproximadamente
  15.48% frente a persistence. En seed 7, VAR(1) gana RMSE y VAR(2) gana NLL
  en la configuración diagonal complementaria, por lo que la historia central
  de drift lineal útil se mantiene.
- Calibración: las coberturas 90% y 95% no siempre alcanzan el valor nominal,
  especialmente en configuraciones H3N2 precisas. Esto sugiere que `Q` requiere
  calibración cuidadosa mediante regularización, inflación o estructura de
  covarianza.

En síntesis:

- H1N1 es candidato a una SDE de random walk gaussiano o drift lineal muy débil.
- H3N2 es candidato a una SDE lineal con drift dependiente del estado, usando
  VAR(1) o VAR(2) aumentado como punto de partida.

## 7. Limitaciones

- La formulación opera en PCA space, no en el espacio latente completo `R^384`.
- El estado actual son centroides mensuales, no cepas individuales.
- No incluye todavía los funcionales `F_viab` ni `F_escape`.
- No genera secuencias ni valida decodificación de trayectorias.
- No modela explícitamente selección inmune, escape antigénico ni fitness.
- La escala temporal mensual se trata como `Delta t = 1`, lo cual es una
  discretización práctica, no una derivación mecanística.
- La covarianza residual puede reflejar ruido de muestreo, cambios de vigilancia
  o heterogeneidad geográfica, no solo difusión evolutiva.

## 8. Próximo paso

Una ruta razonable es:

1. Ajustar una SDE lineal mínima por subtipo en PCA space.
   - H1N1: random walk gaussiano o drift lineal regularizado muy débil.
   - H3N2: drift lineal VAR(1) o VAR(2) aumentado.
2. Evaluar calibración de `Q` con rolling-origin y criterios predefinidos:
   RMSE, NLL, Mahalanobis y cobertura.
3. Solo después de estabilizar la SDE lineal, explorar drift no lineal en PCA
   space.
4. Incorporar términos de viabilidad y escape:

```math
dx_t =
\left[
\alpha \nabla F_{\mathrm{viab}}(x_t)
+
\beta \nabla F_{\mathrm{escape}}(x_t,\mathcal{H}_t)
\right]dt
+
\Sigma(x_t)dW_t.
```

5. Tratar esa extensión como una hipótesis posterior, no como conclusión ya
   demostrada por los modelos gaussianos actuales.

