package com.example.diabetesapp

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.cardview.widget.CardView

class HomeActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_home)

        val diabetesCard = findViewById<CardView>(R.id.card_diabetes)
        val cancerCard = findViewById<CardView>(R.id.card_cancer)
        val heartCard = findViewById<CardView>(R.id.card_heart)
        diabetesCard.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java)
            intent.putExtra("selected", "Diabetes")
            startActivity(intent)
        }

        cancerCard.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java)
            intent.putExtra("selected", "Breast Cancer")  // ✅ must match strings.xml
            startActivity(intent)
        }

        heartCard.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java)
            intent.putExtra("selected", "Heart disease")  // ✅ exact match
            startActivity(intent)
        }

    }
}
