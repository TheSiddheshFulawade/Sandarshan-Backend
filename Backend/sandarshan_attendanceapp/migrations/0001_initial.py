# Generated by Django 5.1.4 on 2024-12-15 19:36

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('sandarshan_userapp', '0002_employeeuser_guestuser'),
    ]

    operations = [
        migrations.CreateModel(
            name='Attendance',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('check_in_time', models.DateTimeField(auto_now_add=True)),
                ('check_out_time', models.DateTimeField(blank=True, null=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='sandarshan_userapp.employeeuser')),
            ],
        ),
    ]
