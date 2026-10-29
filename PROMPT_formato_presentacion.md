# Prompt: presentaciones HTML con este formato

Plantilla para pegar al inicio de una conversación nueva. Describe el formato, el diseño, los componentes y el
comportamiento de la presentación, independientemente del tema. Cambia lo que va entre corchetes.

Nota: de las skills listadas, en el proyecto del Melate solo se cargó `artifact-design`. Las demás se recomiendan
porque habrían ayudado, pero no se usaron.

```text
Hazme una presentación en HTML sobre [TEMA], para presentar en [LUGAR: clase / junta] el [FECHA] ante [PÚBLICO:
p. ej. compañeros y un profesor, no todos expertos]. El contenido sale de [DATOS / RESULTADOS / NOTAS].
Sigue este formato.

SKILLS Y HERRAMIENTAS A USAR
- Antes de escribir el HTML, carga la skill `artifact-design` y sigue su contrato:
  - <title> corto que sea un nombre (2 a 4 palabras, sin explicación después de dos puntos);
  - colores como variables en :root, redefinidas para tema oscuro con prefers-color-scheme y con [data-theme];
  - body con fondo explícito;
  - márgenes laterales max(16px, env(safe-area-inset-*));
  - librerías externas solo de CDNs permitidos y con versión exacta (en este formato no se usa ninguna);
  - sin desborde horizontal a 400 px de ancho.
- Si además quieres que la página guarde datos, comentarios o votos de quien la ve, carga antes la skill
  `artifact-capabilities`.
- Para graficar con buen criterio (forma de la gráfica, paleta validada, leyendas, ejes), carga la skill `dataviz`
  antes de escribir la primera gráfica.
- Para la dirección visual (paleta, tipografía, layout propios del tema, no genéricos), puedes apoyarte en
  `frontend-design:frontend-design` o en `ui-ux-pro-max:slides`.
- Publica con la herramienta Artifact:
  - La primera vez con un ícono genérico (p. ej. "chart") y una descripción de una línea.
  - Después, siempre con el mismo archivo, para conservar el enlace.
  - La versión corta, como archivo y enlace aparte.
- Revisión en navegador con Playwright (scripts propios): capturas por diapositiva en 3 tamaños y 2 temas;
  errores de JavaScript; desbordes; que todo quepa; recorrido completo con las flechas; retroceso con la rueda;
  cuadros lentos al cambiar de diapositiva (con la CPU simulada más lenta).
- Usa subagentes independientes, cada uno con instrucciones de SOLO LECTURA:
  - un revisor de código: navegación, animaciones, casos borde de los datos, accesibilidad y rendimiento;
  - un auditor de UI/UX que revisa las capturas;
  - un auditor de redacción que coteja cada frase contra los datos;
  - si hay cifras calculadas, un agente que las recalcule desde cero sin leer tu código.
  Lánzalos en paralelo y en segundo plano, y verifica sus hallazgos antes de corregir.

ARCHIVO Y DATOS
- Un solo archivo HTML autocontenido que funcione sin internet (sin internet solo cambian las fuentes). Nada de
  librerías: las gráficas se dibujan en SVG con JavaScript propio.
- Ningún número escrito a mano en el texto. Todos los datos van en un bloque <script type="application/json">
  que inserta un script ensamblador (plantilla.html + datos.json → presentación). Títulos, cifras y pies de
  gráfica se arman desde esos datos. Si una frase depende del resultado (por ejemplo, "le gana" o "no le gana"),
  se escribe condicional.
- Dos salidas del mismo ensamblador: el archivo local (con doctype, <html lang="es"> y meta charset utf-8 en los
  primeros 1024 bytes) y una versión sin ese esqueleto, para publicarla como artifact.

ESTRUCTURA DE CADA DIAPOSITIVA
- Cada diapositiva es un <section class="slide"> de pantalla completa (min-height 100vh) con un contenedor .wrap
  de hasta ~1560 px de ancho y márgenes laterales max(16px, 3vw).
- Anatomía fija, en este orden:
  1. eyebrow: etiqueta pequeña en mayúsculas y monoespaciada que dice la sección ("Estadística · ...").
  2. Título (h2) que dice la CONCLUSIÓN de la diapositiva, no el tema. Por ejemplo, "Ninguna prueba encuentra
     memoria", no "Pruebas estadísticas".
  3. Opcional: un recuadro "en palabras simples" (.plain, con borde izquierdo de color) que explica el concepto
     a quien no es experto.
  4. Tarjetas de datos (.facts): número grande arriba, explicación corta abajo, con una línea de color arriba
     que indica de qué tipo de dato se trata.
  5. Una figura (.figure): una tarjeta con la gráfica, su leyenda y un pie de gráfica de 1 o 2 líneas.
- Los títulos quedan a la misma altura en todas las diapositivas (contenido alineado arriba). Solo la portada y
  la diapositiva de la conclusión principal van centradas.
- Columnas con grid y auto-fit que se apilan en pantallas angostas. Nunca desborde horizontal: lo ancho (tablas,
  gráficas) va dentro de su propio contenedor con scroll.
- Diapositivas típicas: portada con un elemento animado del tema; el reto; los datos; el análisis (una diapositiva
  por idea); los resultados; la conclusión en grande; las conclusiones (3 o 4 puntos); un apéndice con glosario y
  material de respaldo.

DISEÑO
- Todos los colores como variables CSS en :root, con tema claro y oscuro (prefers-color-scheme, más un botón que
  cambia el tema y lo recuerda en localStorage dentro de un try/catch).
- Paleta con significado fijo en toda la presentación. Ejemplo: naranja = nuestra respuesta o el elemento clave;
  morado = modelos; verde azulado = teoría o línea base; azul = datos reales; gris = secundario. Ningún color
  cambia de significado entre diapositivas.
- Tipografía: una display con carácter para títulos y números grandes (p. ej. Unbounded), una de texto muy
  legible y sin ceros tachados (p. ej. Source Sans 3), y una monoespaciada para cifras y etiquetas de gráficas
  (p. ej. IBM Plex Mono). Las tres de Google Fonts, con fuentes de respaldo del sistema.
- Para proyector: texto base de ~19–22 px a 1920 de ancho, textos de gráficas de 15 px (en unidades del viewBox),
  gris de texto secundario oscuro y buen contraste. El tema claro es el recomendado para proyectar.
- Gráficas: SVG con viewBox, ejes y etiquetas con los colores del tema, y títulos de eje que no se encimen con
  las marcas. Sin etiquetas cortadas ni encimadas. Si un valor extremo aplasta la escala, se corta el eje y se
  marca "fuera de la gráfica". El eje empieza en cero o lo dice.

MOVIMIENTO
- Todo se ve completo en reposo, sin movimiento (lo que vería una miniatura). Al entrar a una diapositiva por
  PRIMERA vez se reproduce su animación: barras que crecen, líneas que se dibujan, puntos que aparecen
  escalonados, tarjetas que suben.
- Cada diapositiva se anima una sola vez. Los grupos grandes (más de ~60 puntos o barras) se animan como un solo
  bloque, para que no se trabe al cambiar de diapositiva.
- Un momento especial en la portada relacionado con el tema (por ejemplo, unas bolas que salen y se ordenan).
- Respeta prefers-reduced-motion: sin animaciones.

INTERACTIVIDAD (donde sume)
- Un control deslizante para explorar una variable, que en reposo muestra la respuesta correcta.
- Una simulación en vivo con un botón (que diga "Simulación", no use fechas reales).
- Pestañas o un menú para cambiar entre variantes de una misma gráfica.
- Al soltar un control, el foco sale de él para que las flechas vuelvan a cambiar de diapositiva.

NAVEGACIÓN
- Sin scroll-snap: la rueda del mouse se mueve libre en los dos sentidos.
- Flechas, AvPág/RePág, barra espaciadora (con Shift retrocede), Inicio y Fin. Si la diapositiva actual tiene
  contenido abajo (o arriba), primero se recorre y luego se cambia de diapositiva. En la primera y la última no
  da vueltas.
- No captura teclas mientras se usa un control (input, select), ni la barra espaciadora sobre un botón, ni
  combinaciones con Alt/Ctrl.
- Barra de progreso arriba y, abajo a la derecha, contador "n / total" con botones de subir, bajar, tema y
  pantalla completa (este último se oculta si el navegador no la permite).
- La diapositiva actual es la que cruza la línea media de la pantalla, calculada en el scroll. No uses
  IntersectionObserver con rootMargin, porque dentro de un iframe de otro origen no funciona.
- En computadora (≥ 900 px), si una diapositiva es más alta que la pantalla, se reduce con CSS zoom (hasta 70 %)
  hasta que quepa: se mide, se aplica y se vuelve a medir lo que de verdad se ve. En teléfono no se reduce.

REDACCIÓN
- Para no expertos: cada término técnico se explica la primera vez que aparece y va en el glosario.
- Un solo nombre para cada cosa en toda la presentación.
- Pies de gráfica de 1 o 2 líneas que dicen cómo leerla y qué concluir.
- Ninguna frase dice más de lo que muestran los datos: por cada cifra, nombra de dónde sale y revisa que el
  verbo corresponda.
- Meta de densidad: ~60–110 palabras por diapositiva (sin contar gráficas). Lo que no cabe va al apéndice.

ANTES DE ENTREGAR
- Revísala en un navegador real (Playwright) a 1366×768, 1440×900 y 1920×1080, en claro y oscuro, y a 400 px de
  ancho:
  - sin errores de JavaScript ni desbordes horizontales;
  - todas caben;
  - la flecha recorre todas en orden sin saltarse ninguna;
  - la rueda del mouse retrocede;
  - charset UTF-8 y modo estándar;
  - mide los cuadros lentos al cambiar de diapositiva.
- Toma capturas de cada diapositiva y corre dos auditorías independientes (UI/UX y redacción). Verifica sus
  hallazgos contra los datos y corrige.
- Si te pido una versión corta, hazla como copia aparte, sin modificar la original: mismas piezas, menos
  diapositivas, y lo esencial de las que salen en una línea dentro de otras.
```

## Notas

- Funciona igual para otros temas (un reporte, una exposición de clase, una junta): basta cambiar el tema, el
  público y los datos.
- Si no tienes datos en JSON (por ejemplo, una exposición sin análisis), cambia la sección "Archivo y datos" por
  "el contenido está en estas notas" y quita la regla de "ningún número escrito a mano".
- Ejemplo terminado de este formato: `presentacion/plantilla.html` (17 diapositivas) y
  `presentacion/plantilla_corta.html` (11), en esta misma carpeta.
