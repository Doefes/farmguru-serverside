<?php

class MobileController extends \BaseController {

	public function diagnose(){
		$img = Input::file('image');
		$destination = public_path().'/uploads/';
		$filename = str_random(12);

		$tempFile = $img->move($destination, $filename);
		$diagnosis = exec('python '.public_path().'/plantcare/diagnose.py '.escapeshellarg($tempFile));
		File::delete($tempFile);
		if ($diagnosis){
			return $diagnosis;
		}
		else {
			return "Something went wrong!";
		}
	}
}


