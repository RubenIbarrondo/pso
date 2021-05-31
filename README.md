# pso

Resumen esquemático del contenido del repositorio. Se realiza el desarrollo de un algoritmo RR-PSO y se testea su funcionamiento con funciones benchmark. Finalmente, se utiliza para ajustar olas de contagios usando modelos poblacionales. 

## Scripts .py

  * `rr_pso`: Contiene la función que lleva a cabo el algoritmo RR-PSO, además de las subtrutinas necesarias para su funcionamiento.  
    
  * `training_rr_pso`: Contiene las funciones de Griewank y Rosenbrock, y la subrutina para poner a prueba el algoritmo RR-PSO.
  
  * `covid_rr_pso`: Contienene las funciones para realizar ajuste de datos basados en curvas de Verhulst y Gompertz.
  
  * `spectral_radius`: Ejecutable para crear las figuras en las que se muestra el radio espectral en el análisis de estabilidad.

## Notebooks .ipynb
   
   * `Graficos_training`: Describe la obtención de los resultados del entrenamiento de prueba usando las funciones de Griewank y Rosenbrock.
   
   * `Obteniendo_las_series`: Describe el análisis de los datos y la selección de las olas de contagios.
   
   * `Ajuste_covid`: Describe la obtenicón de los ajustes de cada ola, utilizando las funciones definidas en `covid_rr_pso.py`.

    
## Archivos de datos

   * `covid_Spain_25-January-2021.xlsx`: Tabla de datos original, con los datos epidemiológicos de todas las CCAA.
   
   * `covid_aragon.csv`: Selección de los datos epidemiológicos de Aragón.
   
   * `covid_aragon_ola_1`, `covid_aragon_ola_2`, `covid_aragon_ola_3`, y `covid_aragon_ola_4`: Serie de PCR acumulativas separada por olas de contagios.