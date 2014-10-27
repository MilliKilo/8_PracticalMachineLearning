shinyUI(
  pageWithSidebar(
  # Application title
  headerPanel("Predict your Dream vehicle's MPG"),
  
  sidebarPanel(
    sliderInput('variable1', 'Cylinders', value = 4, min = 2, max = 8, step = 2,),
    sliderInput('variable2', 'Displacement', value=72, min = 71.1, max = 472.0, step = 1,),
    sliderInput('variable3', 'Horsepower', value = 90, min = 52.0, max = 335.0, step = 1,),
    sliderInput('variable4', 'Weight', value = 2.0, min = 1.513, max = 5.424, step = .1,),
    sliderInput('variable5', 'Automatic/Manual', value = 0, min = 0, max = 1, step = 1,),
    
    hr(),
    submitButton('Submit')
  ),
  mainPanel(
    h4('Predicted MPG'),
    verbatimTextOutput("prediction")
  )
))