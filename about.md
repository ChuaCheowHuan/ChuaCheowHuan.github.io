---
layout: default
title:

---

<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: Arial;
  font-size: 17px;
}

#myVideo {
  position: fixed;
  left: 0;
  top: 5;
  min-width: 100%;
  min-height: 100%;
}

.content {
  position: fixed;
  top: 50;
  background: rgba(0, 0, 0, 0.5);
  color: #f1f1f1;
  width: 100%;
  padding: 20px;
}

</style>
</head>
<body>

<video autoplay muted loop id="myVideo">
  <source src="/assets/movie/forest_fly.mp4" type="video/mp4">
  Your browser does not support HTML5 video.
</video>

<div class="content">
  <h1>
  </h1>
  <p>
  This site documents my codes in my Github <a href="https://github.com/ChuaCheowHuan">repositories</a>.
  I'm interested in artificial intelligence, specifically reinforcement learning.
  <br><br>
  </p>
</div>

</body>
</html>
