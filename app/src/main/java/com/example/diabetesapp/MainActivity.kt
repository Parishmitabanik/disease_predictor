package com.example.diabetesapp

import android.app.AlertDialog
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    private lateinit var diabetesInterpreter: Interpreter
    private lateinit var cancerInterpreter: Interpreter
    private lateinit var heartInterpreter: Interpreter
    private var currentCancerStep = 1

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "Prediction"

        diabetesInterpreter = Interpreter(loadModelFile("diabetes_model.tflite"))
        cancerInterpreter = Interpreter(loadModelFile("breast_cancer_model.tflite"))
        heartInterpreter = Interpreter(loadModelFile("heart_disease_model.tflite"))
        val selectedFromIntent = intent.getStringExtra("selected")
        val spinner = findViewById<Spinner>(R.id.disease_selector)

        if (selectedFromIntent != null) {
            val position = resources.getStringArray(R.array.disease_options).indexOf(selectedFromIntent)
            if (position != -1) {
                spinner.setSelection(position)
            }
        }

        //val spinner = findViewById<Spinner>(R.id.disease_selector)
        val diabetesLayout = findViewById<LinearLayout>(R.id.diabetes_input_layout)
        val cancerLayout = findViewById<LinearLayout>(R.id.cancer_input_layout)
        val heartLayout = findViewById<LinearLayout>(R.id.heart_input_layout)
        val resultText = findViewById<TextView>(R.id.text_result)
        val predictBtn = findViewById<Button>(R.id.btn_predict)
        val autofillBtn = findViewById<Button>(R.id.btn_autofill)
        val btnNext = findViewById<Button>(R.id.btn_next_step)
        val btnPrev = findViewById<Button>(R.id.btn_prev_step)

        val cancerStep1 = findViewById<LinearLayout>(R.id.cancer_step1)
        val cancerStep2 = findViewById<LinearLayout>(R.id.cancer_step2)
        val cancerStep3 = findViewById<LinearLayout>(R.id.cancer_step3)

        spinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>, view: View?, pos: Int, id: Long) {
                diabetesLayout.visibility = View.GONE
                cancerLayout.visibility = View.GONE
                heartLayout.visibility = View.GONE
                predictBtn.visibility = View.VISIBLE
                autofillBtn.visibility = View.VISIBLE

                when (pos) {
                    0 -> diabetesLayout.visibility = View.VISIBLE
                    1 -> {
                        cancerLayout.visibility = View.VISIBLE
                        currentCancerStep = 1
                        updateCancerStepVisibility(cancerStep1, cancerStep2, cancerStep3, btnPrev, btnNext)
                    }
                    2 -> heartLayout.visibility = View.VISIBLE
                }

                resultText.text = ""
            }

            override fun onNothingSelected(parent: AdapterView<*>) {}
        }

        btnNext.setOnClickListener {
            if (currentCancerStep < 3) {
                currentCancerStep++
                updateCancerStepVisibility(cancerStep1, cancerStep2, cancerStep3, btnPrev, btnNext)
            }
        }

        btnPrev.setOnClickListener {
            if (currentCancerStep > 1) {
                currentCancerStep--
                updateCancerStepVisibility(cancerStep1, cancerStep2, cancerStep3, btnPrev, btnNext)
            }
        }

        predictBtn.setOnClickListener {
            val selected = spinner.selectedItem.toString()

            if (selected == "Diabetes") {
                val inputFields = listOf(
                    R.id.input_pregnancies, R.id.input_glucose, R.id.input_bp, R.id.input_skin,
                    R.id.input_insulin, R.id.input_bmi, R.id.input_dpf, R.id.input_age
                )

                val emptyField = inputFields.firstOrNull { id ->
                    findViewById<EditText>(id).text.toString().isBlank()
                }
                if (emptyField != null) {
                    Toast.makeText(this, "Please fill all diabetes fields.", Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }

                val input = FloatArray(inputFields.size) {
                    findViewById<EditText>(inputFields[it]).text.toString().toFloatOrNull() ?: 0f
                }

                val standardized = standardizeDiabetesInput(input)
                val prediction = predictWithModel(diabetesInterpreter, standardized)
                resultText.text = if (prediction == 1) "Diabetic" else "Not Diabetic"

            } else if (selected == "Heart disease") {
                val heartIds = listOf(
                    R.id.input_age2, R.id.input_sex, R.id.input_cp, R.id.input_trestbps,
                    R.id.input_chol, R.id.input_fbs, R.id.input_restecg, R.id.input_thalach,
                    R.id.input_exang, R.id.input_oldpeak, R.id.input_slope, R.id.input_ca,
                    R.id.input_thal
                )

                val empty = heartIds.firstOrNull {
                    findViewById<EditText>(it).text.toString().isBlank()
                }

                if (empty != null) {
                    Toast.makeText(this, "Please fill all heart fields", Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }

                val input = FloatArray(heartIds.size) { i ->
                    findViewById<EditText>(heartIds[i]).text.toString().toFloatOrNull() ?: 0f
                }

                val prediction = predictWithModel(heartInterpreter, input)
                resultText.text = if (prediction == 1) "Heart Disease" else "Healthy"

            } else {
                val inputIds = arrayOf(
                    R.id.input_radius_mean, R.id.input_texture_mean, R.id.input_perimeter_mean, R.id.input_area_mean,
                    R.id.input_smoothness_mean, R.id.input_compactness_mean, R.id.input_concavity_mean, R.id.input_concave_points_mean,
                    R.id.input_symmetry_mean, R.id.input_fractal_dimension_mean,
                    R.id.input_radius_se, R.id.input_texture_se, R.id.input_perimeter_se, R.id.input_area_se,
                    R.id.input_smoothness_se, R.id.input_compactness_se, R.id.input_concavity_se, R.id.input_concave_points_se,
                    R.id.input_symmetry_se, R.id.input_fractal_dimension_se,
                    R.id.input_radius_worst, R.id.input_texture_worst, R.id.input_perimeter_worst, R.id.input_area_worst,
                    R.id.input_smoothness_worst, R.id.input_compactness_worst, R.id.input_concavity_worst, R.id.input_concave_points_worst,
                    R.id.input_symmetry_worst, R.id.input_fractal_dimension_worst
                )

                val emptyField = inputIds.firstOrNull { id ->
                    findViewById<EditText>(id).text.toString().isBlank()
                }
                if (emptyField != null) {
                    Toast.makeText(this, "Please fill all cancer fields.", Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }

                val input = FloatArray(inputIds.size) { i ->
                    findViewById<EditText>(inputIds[i]).text.toString().toFloatOrNull() ?: 0f
                }

                val prediction = predictWithModel(cancerInterpreter, input)
                resultText.text = if (prediction == 1) "Malignant" else "Benign"
            }
        }

        autofillBtn.setOnClickListener {
            val selected = spinner.selectedItem.toString()

            if (selected == "Diabetes") {
                val values = listOf("3", "120", "72", "22", "85", "33.6", "0.47", "30")
                val inputs = listOf(
                    R.id.input_pregnancies, R.id.input_glucose, R.id.input_bp, R.id.input_skin,
                    R.id.input_insulin, R.id.input_bmi, R.id.input_dpf, R.id.input_age
                )
                inputs.forEachIndexed { i, id ->
                    findViewById<EditText>(id).setText(values[i])
                }
            } else if (selected == "Heart disease") {
                val values = listOf("63", "1", "3", "145", "233", "1", "0", "150", "0", "2.3", "0", "0", "1")
                val inputs = listOf(
                    R.id.input_age2, R.id.input_sex, R.id.input_cp, R.id.input_trestbps,
                    R.id.input_chol, R.id.input_fbs, R.id.input_restecg, R.id.input_thalach,
                    R.id.input_exang, R.id.input_oldpeak, R.id.input_slope, R.id.input_ca,
                    R.id.input_thal
                )
                inputs.forEachIndexed { i, id ->
                    findViewById<EditText>(id).setText(values[i])
                }
            } else {
                val values = (1..30).map { (10..100).random().toString() }
                val cancerIds = arrayOf(
                    R.id.input_radius_mean, R.id.input_texture_mean, R.id.input_perimeter_mean, R.id.input_area_mean,
                    R.id.input_smoothness_mean, R.id.input_compactness_mean, R.id.input_concavity_mean, R.id.input_concave_points_mean,
                    R.id.input_symmetry_mean, R.id.input_fractal_dimension_mean,
                    R.id.input_radius_se, R.id.input_texture_se, R.id.input_perimeter_se, R.id.input_area_se,
                    R.id.input_smoothness_se, R.id.input_compactness_se, R.id.input_concavity_se, R.id.input_concave_points_se,
                    R.id.input_symmetry_se, R.id.input_fractal_dimension_se,
                    R.id.input_radius_worst, R.id.input_texture_worst, R.id.input_perimeter_worst, R.id.input_area_worst,
                    R.id.input_smoothness_worst, R.id.input_compactness_worst, R.id.input_concavity_worst, R.id.input_concave_points_worst,
                    R.id.input_symmetry_worst, R.id.input_fractal_dimension_worst
                )
                cancerIds.forEachIndexed { i, id ->
                    findViewById<EditText>(id).setText(values[i])
                }
            }
        }
    }

    override fun onSupportNavigateUp(): Boolean {
        onBackPressed()
        return true
    }

    private fun predictWithModel(interpreter: Interpreter, input: FloatArray): Int {
        val inputBuffer = ByteBuffer.allocateDirect(4 * input.size).order(ByteOrder.nativeOrder())
        input.forEach { inputBuffer.putFloat(it) }
        inputBuffer.rewind()

        val outputBuffer = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder())
        outputBuffer.rewind()

        interpreter.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()
        val output = outputBuffer.float

        return if (output > 0.5) 1 else 0
    }

    private fun standardizeDiabetesInput(input: FloatArray): FloatArray {
        val means = floatArrayOf(3.8f, 120.9f, 69.1f, 20.5f, 79.8f, 31.9f, 0.47f, 33.2f)
        val stds = floatArrayOf(3.4f, 31.9f, 19.3f, 15.9f, 115.2f, 7.8f, 0.33f, 11.8f)
        return FloatArray(input.size) { i -> (input[i] - means[i]) / stds[i] }
    }

    private fun updateCancerStepVisibility(
        step1: LinearLayout,
        step2: LinearLayout,
        step3: LinearLayout,
        btnPrev: Button,
        btnNext: Button
    ) {
        step1.visibility = if (currentCancerStep == 1) View.VISIBLE else View.GONE
        step2.visibility = if (currentCancerStep == 2) View.VISIBLE else View.GONE
        step3.visibility = if (currentCancerStep == 3) View.VISIBLE else View.GONE

        btnPrev.visibility = if (currentCancerStep > 1) View.VISIBLE else View.GONE
        btnNext.visibility = if (currentCancerStep < 3) View.VISIBLE else View.GONE
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        menuInflater.inflate(R.menu.menu_main, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.menu_about -> {
                AlertDialog.Builder(this)
                    .setTitle("About")
                    .setMessage("Harnessing machine learning to assess your risk of Diabetes, Breast Cancer, and Heart Disease — quickly and intelligently.\n\nDeveloped by Parishmita.")
                    .setPositiveButton("OK", null)
                    .show()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }
}
