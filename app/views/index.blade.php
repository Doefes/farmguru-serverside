<!doctype html>
<html lang="en">
<head>
	<meta charset="UTF-8">
	<title>Plantcare Backend</title>
	<style>
		@import url(//fonts.googleapis.com/css?family=Lato:700);

		body {
			margin:0;
			font-family:'Lato', sans-serif;
			text-align:center;
			color: #999;
			background-color: #999;
		}

		.welcome {
			width: 300px;
			height: 200px;
			position: absolute;
			left: 50%;
			top: 40%;
			margin-left: -150px;
			margin-top: -100px;
		}

		a, a:visited {
			text-decoration:none;
		}

		h1 {
			font-size: 32px;
			margin: 30px 0 0 0;
			color: white;
		}
	</style>
</head>
<body>
	<div class="welcome">
		<a href="http://project.cmi.hro.nl/2014_2015/nll_mt2c_t3/" title="Plantcare Project MT"><img src="{{URL::asset('assets/img/logo.png')}}" alt="Plantcare Project MT"></a>
		<h1>Plantcare Backend</h1>
	</div>
</body>
</html>
