$(function () {
  
function q(selector) {return document.querySelector(selector)}
  
$(function(){
  $("#button").click(function(){
      // 텍스트 빈 값 확인
      if(!document.sentMessage.text.value){
        alert("내용을 입력하세요");
        document.sentMessage.text.focus()
        return false
      }
        // Synthesize 처리과정 노출
        text = q('#text').value.trim()

        if (text) {
          q('#message').textContent = 'Synthesizing...'
          q('#button').disabled = true
          q('#audio').hidden = true
        }

      // 텍스트 값 전달
      urls = '/tts/synthesize/';
      $.ajax({        
        url : urls,
        datatype :"text",
        type :"POST",
        data : {"text":$("#text").val()},
        success : function(result){
            audio = result["audio"] 
            //console.log(audio);
            //alert(audio);
            q('#audio').hidden = false
            q('#audio').src = audio
        }
      }); 
  });
});

});