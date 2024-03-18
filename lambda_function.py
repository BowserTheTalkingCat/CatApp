import boto3
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read
from scipy.signal import spectrogram
from io import BytesIO

def lambda_handler(event, context):
    # Client to access S3
    s3_client = boto3.client('s3')
    
    # Details of the source audio file
    source_bucket_name = event['source_bucket_name']
    source_object_key = event['source_object_key']
    
    # Details for saving the spectrogram
    target_bucket_name = event['target_bucket_name']
    target_object_key = event['target_object_key']

    # Fetch the audio file from S3
    audio_data = fetch_audio_from_s3(s3_client, source_bucket_name, source_object_key)

    # Generate a spectrogram
    spectrogram_image = generate_spectrogram(audio_data)

    # Save the spectrogram image to S3
    save_spectrogram_to_s3(s3_client, spectrogram_image, target_bucket_name, target_object_key)
    
    return {
        'statusCode': 200,
        'body': 'Spectrogram generated and saved successfully.'
    }

def fetch_audio_from_s3(s3_client, bucket_name, object_key):
    s3_object = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    audio_data = s3_object['Body'].read()
    return BytesIO(audio_data)

def generate_spectrogram(audio_data):
    samplerate, data = read(audio_data)
    f, t, Sxx = spectrogram(data, samplerate)
    plt.pcolormesh(t, f, np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Log Intensity [dB]')
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

def save_spectrogram_to_s3(s3_client, image_data, bucket_name, object_key):
    s3_client.upload_fileobj(image_data, bucket_name, object_key)

