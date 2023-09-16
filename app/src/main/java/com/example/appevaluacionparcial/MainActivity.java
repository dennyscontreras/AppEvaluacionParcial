package com.example.appevaluacionparcial;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import com.example.appevaluacionparcial.ml.Mymodel;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {
    Button selectbtn,capturabtn,predictbtn;
    TextView result;
    ImageView imageView;
    Bitmap bitmap;
    String[] labels = {"Dennys Contreras", "Rafael Correa"};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Permisos
        getPermission();

        selectbtn=findViewById(R.id.selectbtn);
        capturabtn=findViewById(R.id.capturabtn);
        predictbtn=findViewById(R.id.predictbtn);
        result=findViewById(R.id.result);
        imageView=findViewById(R.id.imageView);
        selectbtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent=new Intent();
                intent.setAction(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent,10);

            }
        });
        capturabtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent=new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent,12);

            }
        });
        predictbtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (bitmap != null) {
                    try {
                        Mymodel model = Mymodel.newInstance(MainActivity.this);

                        // Creates inputs for reference.
                        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
                        bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
                       // inputFeature0.loadBuffer(TensorImage.fromBitmap(bitmap).getBuffer());

                        // Runs model inference and gets result.


                        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3);
                        byteBuffer.order(ByteOrder.nativeOrder());

                        // pixeles de imagen
                        int [] intValues = new int[224 * 224];
                        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

                        // normalizar la imagen en el bytebuffer.
                        int pixel = 0;
                        for(int i = 0; i < 224; i++){
                            for(int j = 0; j < 224; j++){
                                int val = intValues[pixel++];
                                byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                                byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                                byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                            }
                        }

                        inputFeature0.loadBuffer(byteBuffer);
                        Mymodel.Outputs outputs = model.process(inputFeature0);
                        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                        result.setText(labels[getMax(outputFeature0.getFloatArray())] + " ");

                        // Releases model resources if no longer used.
                        model.close();
                    } catch (IOException e) {
                        // TODO Handle the exception
                        e.printStackTrace();
                    }
                } else {
                    // Manejar el caso en el que bitmap es nulo (por ejemplo, no se ha seleccionado una imagen).
                    // Puedes mostrar un mensaje de error o tomar alguna otra acción apropiada.
                    result.setText("Por favor, selecciona una imagen antes de predecir.");
                }
            }
        });
    }

    int getMax(float[] arr) {
        int max = 0;
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] > arr[max]) max = i;
        }
        return max;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == 11) {
            if (grantResults.length > 0) {
                if (grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                    this.getPermission();
                }
            }
        }

        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    void getPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA}, 11);
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == 10) {
            if (data != null) {
                Uri uri = data.getData();
                try {
                    bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                    imageView.setImageBitmap(bitmap);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        } else if (requestCode == 12) {
            bitmap = (Bitmap) data.getExtras().get("data");
            imageView.setImageBitmap(bitmap);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}
