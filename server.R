library(caret)
library(shiny)
data(mtcars)

fit <- train(mpg ~ cyl+disp+hp+wt+am, method = "glm", data = mtcars)

result <- function(cyl,disp,hp,wt,am) {
  s <- cbind(cyl,disp,hp,wt,am)
  res <- predict(fit,s)
  as.character(res)
}

shinyServer(
  function(input, output) {
    output$prediction <- renderPrint({result(input$variable1,input$variable2,input$variable3,input$variable4,input$variable5)})
  }
)